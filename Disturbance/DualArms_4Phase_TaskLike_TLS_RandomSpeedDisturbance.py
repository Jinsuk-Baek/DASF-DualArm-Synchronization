import csv
import math
import random
import threading
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

from xarm import XArmAPI


# ============================================================
# Experiment mode
# ============================================================
RT_MLS_ENABLED = False
EXPERIMENT_LABEL = "TLS-RANDOM-SPEED-DISTURBANCE"

ARM_A_IP = "192.168.0.251"
ARM_B_IP = "192.168.0.252"

# ============================================================
# Robot-motion parameters
# ============================================================
BASE_CART_SPEED = 80.0       # mm/s
JOINT_SPEED = 20.0           # deg/s
JOINT_ACC = 200.0            # deg/s^2

CONTROL_INTERVAL_SEC = 0.05  # 20-Hz velocity update
VELOCITY_COMMAND_DURATION_SEC = 0.10
POSITION_TOLERANCE_MM = 1.5
FINAL_CONFIRM_SAMPLES = 2
APPROACH_ZONE_MM = 25.0
MIN_APPROACH_SPEED_MM_S = 8.0

BARRIER_TIMEOUT_SEC = 5.0
THREAD_JOIN_TIMEOUT_SEC = 35.0

# ============================================================
# RT-MLS parameters
# ============================================================
MLS_PROGRESS_DEADBAND = 0.010
MLS_KP = 0.35
MLS_MAX_STEP_CORRECTION = 0.025
MLS_MIN_SCALE = 0.65
MLS_MAX_SCALE = 1.20
MLS_RELAXATION = 0.02

# ============================================================
# Intermittent random-disturbance parameters
#
# A disturbance is injected into approximately 20% of coordinated
# phase executions. During the selected phase, one randomly selected
# arm is temporarily limited to 40-70% of its current commanded speed
# for 200-500 ms after reaching a random point between 20% and 60%
# of its path.
#
# The speed degradation is multiplicative:
#
# actual_speed = BASE_CART_SPEED * RT_MLS_scale * disturbance_scale
#
# Thus, TLS remains at scale=1.0 while TLS+RT-MLS can compensate by
# modifying its real-time command scale during the disturbance.
#
# The seed and deterministic per-phase generation guarantee that the
# TLS and TLS+RT-MLS scripts use exactly the same disturbance schedule.
# ============================================================
DISTURBANCE_PROBABILITY = 0.20
DISTURBANCE_MIN_SEC = 0.20
DISTURBANCE_MAX_SEC = 0.50
DISTURBANCE_SPEED_SCALE_MIN = 0.40
DISTURBANCE_SPEED_SCALE_MAX = 0.70
DISTURBANCE_TRIGGER_MIN_PROGRESS = 0.20
DISTURBANCE_TRIGGER_MAX_PROGRESS = 0.60
DISTURBANCE_RANDOM_SEED = 20260805

NUM_CYCLES = 30
PHASE_HOLD_SEC = 0.20

CSV_PATH = "dual_arm_tls_random_speed_disturbance_log.csv"
DISTURBANCE_PLAN_CSV_PATH = "dual_arm_random_speed_disturbance_plan.csv"

READY_JOINTS_A = [5.0, -20.0, 50.0, 0.0, 10.0, 0.0]
READY_JOINTS_B = [-10.0, -20.0, 50.0, 0.0, 15.0, 0.0]

ARM_B_DX_SIGN = 1.0
ARM_B_DY_SIGN = 1.0
ARM_B_DZ_SIGN = 1.0

TASK_PATTERN: List[Tuple[str, Tuple[float, float, float]]] = [
    ("Approach",   (0.0,   0.0, -30.0)),
    ("Lift",       (0.0,   0.0,  30.0)),
    ("Transfer-1", (70.0,  0.0,  25.0)),
    ("Transfer-2", (95.0, 55.0,  15.0)),
    ("Place",      (60.0, 85.0,   0.0)),
    ("Retreat",    (20.0, 45.0,  25.0)),
    ("Return-1",   (-30.0, 10.0, 20.0)),
    ("Return-2",   (0.0,   0.0,   0.0)),
]


def now() -> float:
    return time.perf_counter()


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def norm3(x: float, y: float, z: float) -> float:
    return math.sqrt(x * x + y * y + z * z)


def normalize3(x: float, y: float, z: float) -> Tuple[float, float, float]:
    length = norm3(x, y, z)
    if length <= 1e-9:
        return 0.0, 0.0, 0.0
    return x / length, y / length, z / length


@dataclass(frozen=True)
class DisturbanceSpec:
    enabled: bool
    arm_key: str
    duration_sec: float
    trigger_progress: float
    speed_scale: float


@dataclass
class ArmTiming:
    dispatch_time: float
    finish_time: float
    exec_ms: float


@dataclass
class ThreadResult:
    success: bool
    timing: Optional[ArmTiming] = None
    error: Optional[str] = None


@dataclass
class SyncLogEntry:
    record_id: int
    cycle_index: int
    phase_index: int
    phase_name: str
    label: str
    experiment: str

    disturbance_enabled: int
    disturbed_arm: str
    disturbance_duration_ms: float
    disturbance_trigger_progress: float
    disturbance_speed_scale: float
    disturbance_applied: int
    disturbance_actual_start_progress: float

    arm_a_init_scale: float
    arm_b_init_scale: float
    rt_scale_a_final: float
    rt_scale_b_final: float
    rt_scale_a_min: float
    rt_scale_a_max: float
    rt_scale_a_mean: float
    rt_scale_b_min: float
    rt_scale_b_max: float
    rt_scale_b_mean: float
    rt_update_count: int
    max_abs_progress_gap: float

    arm_a_dispatch_time: float
    arm_b_dispatch_time: float
    dispatch_gap_ms: float
    arm_a_finish_time: float
    arm_b_finish_time: float
    finish_gap_ms: float
    signed_finish_error_ms: float
    arm_a_exec_ms: float
    arm_b_exec_ms: float

    arm_a_target_x: float
    arm_a_target_y: float
    arm_a_target_z: float
    arm_a_target_roll: float
    arm_a_target_pitch: float
    arm_a_target_yaw: float
    arm_b_target_x: float
    arm_b_target_y: float
    arm_b_target_z: float
    arm_b_target_roll: float
    arm_b_target_pitch: float
    arm_b_target_yaw: float

    status: str
    error_msg: str


def build_disturbance_plan(num_cycles: int) -> Dict[Tuple[int, int], DisturbanceSpec]:
    """
    Build one reproducible plan shared by both scripts.

    A separate deterministic seed is derived for each cycle/phase pair, so
    changing thread timing or execution order cannot change the disturbance.
    """
    plan: Dict[Tuple[int, int], DisturbanceSpec] = {}

    for cycle_index in range(1, num_cycles + 1):
        for phase_index in range(1, len(TASK_PATTERN) + 1):
            local_seed = (
                DISTURBANCE_RANDOM_SEED
                + cycle_index * 1009
                + phase_index * 9176
            )
            rng = random.Random(local_seed)
            enabled = rng.random() < DISTURBANCE_PROBABILITY

            if enabled:
                arm_key = "A" if rng.random() < 0.5 else "B"
                duration_sec = rng.uniform(
                    DISTURBANCE_MIN_SEC,
                    DISTURBANCE_MAX_SEC,
                )
                trigger_progress = rng.uniform(
                    DISTURBANCE_TRIGGER_MIN_PROGRESS,
                    DISTURBANCE_TRIGGER_MAX_PROGRESS,
                )
                speed_scale = rng.uniform(
                    DISTURBANCE_SPEED_SCALE_MIN,
                    DISTURBANCE_SPEED_SCALE_MAX,
                )
            else:
                arm_key = ""
                duration_sec = 0.0
                trigger_progress = 0.0
                speed_scale = 1.0

            plan[(cycle_index, phase_index)] = DisturbanceSpec(
                enabled=enabled,
                arm_key=arm_key,
                duration_sec=duration_sec,
                trigger_progress=trigger_progress,
                speed_scale=speed_scale,
            )

    return plan


def save_disturbance_plan(
    plan: Dict[Tuple[int, int], DisturbanceSpec],
    csv_path: str,
) -> None:
    with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
        fieldnames = [
            "cycle_index",
            "phase_index",
            "phase_name",
            "disturbance_enabled",
            "disturbed_arm",
            "disturbance_duration_ms",
            "disturbance_trigger_progress",
            "disturbance_speed_scale",
        ]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for cycle_index in range(1, NUM_CYCLES + 1):
            for phase_index, (phase_name, _) in enumerate(
                TASK_PATTERN, start=1
            ):
                spec = plan[(cycle_index, phase_index)]
                writer.writerow(
                    {
                        "cycle_index": cycle_index,
                        "phase_index": phase_index,
                        "phase_name": phase_name,
                        "disturbance_enabled": int(spec.enabled),
                        "disturbed_arm": spec.arm_key,
                        "disturbance_duration_ms": spec.duration_sec * 1000.0,
                        "disturbance_trigger_progress": spec.trigger_progress,
                        "disturbance_speed_scale": spec.speed_scale,
                    }
                )


class RobotArm:
    def __init__(self, ip: str, name: str) -> None:
        self.ip = ip
        self.name = name
        self.arm = XArmAPI(ip, do_not_open=False)

    @staticmethod
    def _ret_code(ret) -> int:
        return ret[0] if isinstance(ret, tuple) else ret

    def ensure_ready_position_mode(self) -> None:
        for _ in range(3):
            self.arm.clean_error()
            self.arm.clean_warn()
            self.arm.motion_enable(True)
            self.arm.set_mode(0)
            self.arm.set_state(0)
            time.sleep(0.3)

            if self.arm.error_code == 0 and self.arm.state in (0, 1, 2):
                return

        raise RuntimeError(
            f"[{self.name}] Robot not ready "
            f"(state={self.arm.state}, error={self.arm.error_code}, "
            f"warn={self.arm.warn_code})"
        )

    def enter_cartesian_velocity_mode(self) -> None:
        self.arm.clean_error()
        self.arm.clean_warn()
        self.arm.motion_enable(True)

        ret = self._ret_code(self.arm.set_mode(5))
        if ret != 0:
            raise RuntimeError(f"[{self.name}] set_mode(5) failed, ret={ret}")

        ret = self._ret_code(self.arm.set_state(0))
        if ret != 0:
            raise RuntimeError(f"[{self.name}] set_state(0) failed, ret={ret}")

        time.sleep(0.20)

        if self.arm.error_code != 0:
            raise RuntimeError(
                f"[{self.name}] velocity-mode error={self.arm.error_code}"
            )

    def exit_cartesian_velocity_mode(self) -> None:
        try:
            self.stop_cartesian_velocity()
        except Exception:
            pass
        try:
            self.arm.set_mode(0)
            self.arm.set_state(0)
            time.sleep(0.15)
        except Exception:
            pass

    def connect(self) -> None:
        print(f"[{self.name}] Connecting to {self.ip}...")
        if not self.arm.connected:
            raise RuntimeError(f"[{self.name}] Connection failed")
        self.ensure_ready_position_mode()
        print(f"[{self.name}] Connected successfully")

    def get_pose(self) -> List[float]:
        ret = self.arm.get_position(is_radian=False)
        if not isinstance(ret, tuple) or len(ret) < 2 or ret[0] != 0:
            raise RuntimeError(f"[{self.name}] Failed to read pose: {ret}")
        pose = list(ret[1])
        if len(pose) < 6:
            raise RuntimeError(f"[{self.name}] Invalid pose: {pose}")
        return pose[:6]

    def get_joints(self):
        return self.arm.get_servo_angle(is_radian=False)

    def command_cartesian_velocity(
        self,
        vx: float,
        vy: float,
        vz: float,
    ) -> int:
        ret = self.arm.vc_set_cartesian_velocity(
            [vx, vy, vz, 0.0, 0.0, 0.0],
            is_radian=False,
            is_tool_coord=False,
            duration=VELOCITY_COMMAND_DURATION_SEC,
        )
        return self._ret_code(ret)

    def stop_cartesian_velocity(self) -> None:
        ret = self.command_cartesian_velocity(0.0, 0.0, 0.0)
        if ret != 0:
            raise RuntimeError(
                f"[{self.name}] stop velocity failed, ret={ret}"
            )

    def move_joints_retry(self, joints: List[float], retries: int = 2) -> int:
        last_ret = None
        last_error = None

        for _ in range(retries + 1):
            try:
                self.ensure_ready_position_mode()
                ret = self.arm.set_servo_angle(
                    angle=joints,
                    speed=JOINT_SPEED,
                    mvacc=JOINT_ACC,
                    wait=True,
                    is_radian=False,
                )
                ret = self._ret_code(ret)
                if ret == 0:
                    return ret
                last_ret = ret
            except Exception as exc:
                last_error = str(exc)

            time.sleep(0.3)

        if last_error:
            raise RuntimeError(f"[{self.name}] joint move failed: {last_error}")
        raise RuntimeError(f"[{self.name}] joint move failed, ret={last_ret}")

    def disconnect(self) -> None:
        try:
            self.exit_cartesian_velocity_mode()
        except Exception:
            pass
        try:
            self.arm.disconnect()
        except Exception:
            pass


class SharedMotionState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.stop_event = threading.Event()

        self.scale_a = 1.0
        self.scale_b = 1.0
        self.progress_a = 0.0
        self.progress_b = 0.0
        self.total_distance_a = 0.0
        self.total_distance_b = 0.0
        self.start_a = 0.0
        self.start_b = 0.0
        self.a_done = False
        self.b_done = False

        self.scale_history_a: List[float] = [1.0]
        self.scale_history_b: List[float] = [1.0]
        self.update_count = 0
        self.max_abs_progress_gap = 0.0

        self.disturbance_applied = False
        self.disturbance_actual_start_progress = 0.0


def rt_mls_progress_controller(state: SharedMotionState) -> None:
    """
    TLS: leaves both scales fixed at 1.0.
    TLS+RT-MLS: changes the actual velocity-command scales based on the
    normalized path-progress gap.
    """
    while not state.stop_event.wait(CONTROL_INTERVAL_SEC):
        with state.lock:
            if state.a_done and state.b_done:
                break

            if (
                state.start_a == 0.0
                or state.start_b == 0.0
                or state.total_distance_a <= 0.0
                or state.total_distance_b <= 0.0
            ):
                continue

            progress_gap = state.progress_a - state.progress_b
            state.max_abs_progress_gap = max(
                state.max_abs_progress_gap,
                abs(progress_gap),
            )

            if RT_MLS_ENABLED and not state.a_done and not state.b_done:
                if abs(progress_gap) > MLS_PROGRESS_DEADBAND:
                    correction = clamp(
                        MLS_KP * progress_gap,
                        -MLS_MAX_STEP_CORRECTION,
                        MLS_MAX_STEP_CORRECTION,
                    )

                    # A ahead -> slow A, accelerate B.
                    state.scale_a = clamp(
                        state.scale_a - correction,
                        MLS_MIN_SCALE,
                        MLS_MAX_SCALE,
                    )
                    state.scale_b = clamp(
                        state.scale_b + correction,
                        MLS_MIN_SCALE,
                        MLS_MAX_SCALE,
                    )
                    state.update_count += 1
                else:
                    state.scale_a += MLS_RELAXATION * (1.0 - state.scale_a)
                    state.scale_b += MLS_RELAXATION * (1.0 - state.scale_b)
            else:
                state.scale_a = 1.0
                state.scale_b = 1.0

            state.scale_history_a.append(state.scale_a)
            state.scale_history_b.append(state.scale_b)


def run_arm_velocity_motion(
    arm: RobotArm,
    target_pose: Tuple[float, float, float, float, float, float],
    barrier: threading.Barrier,
    output: Dict[str, ThreadResult],
    key: str,
    state: SharedMotionState,
    disturbance: DisturbanceSpec,
) -> None:
    try:
        start_pose = arm.get_pose()
        target_x, target_y, target_z = target_pose[:3]

        total_distance = norm3(
            target_x - start_pose[0],
            target_y - start_pose[1],
            target_z - start_pose[2],
        )

        with state.lock:
            if key == "A":
                state.total_distance_a = total_distance
            else:
                state.total_distance_b = total_distance

        barrier.wait(timeout=BARRIER_TIMEOUT_SEC)
        arm.enter_cartesian_velocity_mode()
        dispatch_t = now()

        with state.lock:
            if key == "A":
                state.start_a = dispatch_t
            else:
                state.start_b = dispatch_t

        if total_distance <= POSITION_TOLERANCE_MM:
            arm.stop_cartesian_velocity()
            finish_t = now()
            with state.lock:
                if key == "A":
                    state.progress_a = 1.0
                    state.a_done = True
                else:
                    state.progress_b = 1.0
                    state.b_done = True

            output[key] = ThreadResult(
                success=True,
                timing=ArmTiming(
                    dispatch_time=dispatch_t,
                    finish_time=finish_t,
                    exec_ms=(finish_t - dispatch_t) * 1000.0,
                ),
            )
            return

        confirm_count = 0
        local_disturbance_applied = False
        loop_start = now()
        motion_timeout = THREAD_JOIN_TIMEOUT_SEC - 3.0

        while True:
            if now() - loop_start > motion_timeout:
                raise RuntimeError(
                    f"[{arm.name}] Cartesian velocity motion timeout"
                )

            current_pose = arm.get_pose()
            error_x = target_x - current_pose[0]
            error_y = target_y - current_pose[1]
            error_z = target_z - current_pose[2]
            remaining = norm3(error_x, error_y, error_z)
            progress = clamp(
                1.0 - remaining / max(total_distance, 1e-9),
                0.0,
                1.0,
            )

            with state.lock:
                if key == "A":
                    state.progress_a = progress
                    command_scale = state.scale_a
                else:
                    state.progress_b = progress
                    command_scale = state.scale_b

            # Inject one mid-motion disturbance into the selected arm.
            if (
                disturbance.enabled
                and disturbance.arm_key == key
                and not local_disturbance_applied
                and progress >= disturbance.trigger_progress
                and remaining > POSITION_TOLERANCE_MM
            ):
                local_disturbance_applied = True

                with state.lock:
                    state.disturbance_applied = True
                    state.disturbance_actual_start_progress = progress

                print(
                    f"[DISTURBANCE] {arm.name}: temporary speed degradation "
                    f"to {disturbance.speed_scale * 100.0:.1f}% for "
                    f"{disturbance.duration_sec * 1000.0:.1f} ms "
                    f"at progress={progress:.3f}"
                )

                disturbance_end = now() + disturbance.duration_sec

                while now() < disturbance_end:
                    # Re-read the current pose and the latest RT-MLS scale
                    # on every control update. This allows RT-MLS to continue
                    # compensating while the external speed degradation is active.
                    disturbed_pose = arm.get_pose()
                    disturbed_error_x = target_x - disturbed_pose[0]
                    disturbed_error_y = target_y - disturbed_pose[1]
                    disturbed_error_z = target_z - disturbed_pose[2]
                    disturbed_remaining = norm3(
                        disturbed_error_x,
                        disturbed_error_y,
                        disturbed_error_z,
                    )

                    if disturbed_remaining <= POSITION_TOLERANCE_MM:
                        break

                    disturbed_unit_x, disturbed_unit_y, disturbed_unit_z = normalize3(
                        disturbed_error_x,
                        disturbed_error_y,
                        disturbed_error_z,
                    )

                    with state.lock:
                        if key == "A":
                            current_rt_scale = state.scale_a
                        else:
                            current_rt_scale = state.scale_b

                    disturbed_speed = (
                        BASE_CART_SPEED
                        * current_rt_scale
                        * disturbance.speed_scale
                    )

                    if disturbed_remaining < APPROACH_ZONE_MM:
                        approach_speed = BASE_CART_SPEED * (
                            disturbed_remaining / APPROACH_ZONE_MM
                        )
                        disturbed_speed = min(
                            disturbed_speed,
                            max(MIN_APPROACH_SPEED_MM_S, approach_speed),
                        )

                    ret = arm.command_cartesian_velocity(
                        disturbed_unit_x * disturbed_speed,
                        disturbed_unit_y * disturbed_speed,
                        disturbed_unit_z * disturbed_speed,
                    )
                    if ret != 0:
                        raise RuntimeError(
                            f"[{arm.name}] speed disturbance failed, ret={ret}"
                        )

                    time.sleep(CONTROL_INTERVAL_SEC)

                # Re-read pose on the next iteration and resume normal motion.
                continue

            if remaining <= POSITION_TOLERANCE_MM:
                confirm_count += 1
                arm.stop_cartesian_velocity()
                if confirm_count >= FINAL_CONFIRM_SAMPLES:
                    break
                time.sleep(CONTROL_INTERVAL_SEC)
                continue

            confirm_count = 0
            unit_x, unit_y, unit_z = normalize3(
                error_x,
                error_y,
                error_z,
            )
            commanded_speed = BASE_CART_SPEED * command_scale

            if remaining < APPROACH_ZONE_MM:
                approach_speed = BASE_CART_SPEED * (
                    remaining / APPROACH_ZONE_MM
                )
                commanded_speed = min(
                    commanded_speed,
                    max(MIN_APPROACH_SPEED_MM_S, approach_speed),
                )

            ret = arm.command_cartesian_velocity(
                unit_x * commanded_speed,
                unit_y * commanded_speed,
                unit_z * commanded_speed,
            )
            if ret != 0:
                raise RuntimeError(
                    f"[{arm.name}] vc_set_cartesian_velocity failed, ret={ret}"
                )

            time.sleep(CONTROL_INTERVAL_SEC)

        arm.stop_cartesian_velocity()
        finish_t = now()

        with state.lock:
            if key == "A":
                state.progress_a = 1.0
                state.a_done = True
            else:
                state.progress_b = 1.0
                state.b_done = True

        output[key] = ThreadResult(
            success=True,
            timing=ArmTiming(
                dispatch_time=dispatch_t,
                finish_time=finish_t,
                exec_ms=(finish_t - dispatch_t) * 1000.0,
            ),
        )

    except threading.BrokenBarrierError:
        output[key] = ThreadResult(
            success=False,
            error=f"[{arm.name}] barrier broken or timeout",
        )
    except Exception as exc:
        try:
            barrier.abort()
        except Exception:
            pass
        try:
            arm.stop_cartesian_velocity()
        except Exception:
            pass
        output[key] = ThreadResult(
            success=False,
            error=f"[{arm.name}] {exc}",
        )


def save_csv(log_entries: List[SyncLogEntry], csv_path: str) -> None:
    if not log_entries:
        return

    with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=list(asdict(log_entries[0]).keys()),
        )
        writer.writeheader()
        for entry in log_entries:
            writer.writerow(asdict(entry))


def synchronized_cartesian_move(
    arm_a: RobotArm,
    pose_a: Tuple[float, float, float, float, float, float],
    arm_b: RobotArm,
    pose_b: Tuple[float, float, float, float, float, float],
    disturbance: DisturbanceSpec,
    label: str,
    cycle_index: int,
    phase_index: int,
    phase_name: str,
    record_id: int,
    log_entries: List[SyncLogEntry],
) -> int:
    print(f"\n[{EXPERIMENT_LABEL}] {label}")
    if disturbance.enabled:
        print(
            f"[PLAN] disturb Arm-{disturbance.arm_key}, "
            f"duration={disturbance.duration_sec * 1000.0:.1f} ms, "
            f"speed={disturbance.speed_scale:.3f}, "
            f"trigger={disturbance.trigger_progress:.3f}"
        )

    state = SharedMotionState()
    barrier = threading.Barrier(2)
    output: Dict[str, ThreadResult] = {}

    controller_thread = threading.Thread(
        target=rt_mls_progress_controller,
        args=(state,),
        daemon=True,
    )
    thread_a = threading.Thread(
        target=run_arm_velocity_motion,
        args=(arm_a, pose_a, barrier, output, "A", state, disturbance),
        daemon=True,
    )
    thread_b = threading.Thread(
        target=run_arm_velocity_motion,
        args=(arm_b, pose_b, barrier, output, "B", state, disturbance),
        daemon=True,
    )

    controller_thread.start()
    thread_a.start()
    thread_b.start()

    thread_a.join(timeout=THREAD_JOIN_TIMEOUT_SEC)
    thread_b.join(timeout=THREAD_JOIN_TIMEOUT_SEC)

    state.stop_event.set()
    controller_thread.join(timeout=2.0)

    for arm in (arm_a, arm_b):
        try:
            arm.stop_cartesian_velocity()
        except Exception:
            pass
        arm.exit_cartesian_velocity_mode()

    if thread_a.is_alive() or thread_b.is_alive():
        raise RuntimeError(f"Thread timeout during '{label}'")

    if "A" not in output or "B" not in output:
        raise RuntimeError(f"Missing thread output during '{label}'")

    result_a = output["A"]
    result_b = output["B"]

    if not result_a.success or not result_b.success:
        error_a = result_a.error if not result_a.success else ""
        error_b = result_b.error if not result_b.success else ""
        raise RuntimeError(
            f"Motion failed during '{label}': {error_a} {error_b}".strip()
        )

    assert result_a.timing is not None
    assert result_b.timing is not None

    dispatch_gap_ms = abs(
        result_a.timing.dispatch_time - result_b.timing.dispatch_time
    ) * 1000.0
    finish_gap_ms = abs(
        result_a.timing.finish_time - result_b.timing.finish_time
    ) * 1000.0
    signed_finish_error_ms = (
        result_a.timing.finish_time - result_b.timing.finish_time
    ) * 1000.0

    with state.lock:
        history_a = state.scale_history_a.copy()
        history_b = state.scale_history_b.copy()
        update_count = state.update_count
        max_abs_progress_gap = state.max_abs_progress_gap
        disturbance_applied = state.disturbance_applied
        actual_start_progress = state.disturbance_actual_start_progress

    mean_a = sum(history_a) / len(history_a)
    mean_b = sum(history_b) / len(history_b)

    print(f"[RESULT] Dispatch gap : {dispatch_gap_ms:.2f} ms")
    print(f"[RESULT] Finish gap   : {finish_gap_ms:.2f} ms")
    print(
        f"[RESULT] Scale A min/max/final: "
        f"{min(history_a):.3f}/{max(history_a):.3f}/{history_a[-1]:.3f}"
    )
    print(
        f"[RESULT] Scale B min/max/final: "
        f"{min(history_b):.3f}/{max(history_b):.3f}/{history_b[-1]:.3f}"
    )
    print(f"[RESULT] RT updates   : {update_count}")

    log_entries.append(
        SyncLogEntry(
            record_id=record_id,
            cycle_index=cycle_index,
            phase_index=phase_index,
            phase_name=phase_name,
            label=label,
            experiment=EXPERIMENT_LABEL,

            disturbance_enabled=int(disturbance.enabled),
            disturbed_arm=disturbance.arm_key,
            disturbance_duration_ms=disturbance.duration_sec * 1000.0,
            disturbance_trigger_progress=disturbance.trigger_progress,
            disturbance_speed_scale=disturbance.speed_scale,
            disturbance_applied=int(disturbance_applied),
            disturbance_actual_start_progress=actual_start_progress,

            arm_a_init_scale=1.0,
            arm_b_init_scale=1.0,
            rt_scale_a_final=history_a[-1],
            rt_scale_b_final=history_b[-1],
            rt_scale_a_min=min(history_a),
            rt_scale_a_max=max(history_a),
            rt_scale_a_mean=mean_a,
            rt_scale_b_min=min(history_b),
            rt_scale_b_max=max(history_b),
            rt_scale_b_mean=mean_b,
            rt_update_count=update_count,
            max_abs_progress_gap=max_abs_progress_gap,

            arm_a_dispatch_time=result_a.timing.dispatch_time,
            arm_b_dispatch_time=result_b.timing.dispatch_time,
            dispatch_gap_ms=dispatch_gap_ms,
            arm_a_finish_time=result_a.timing.finish_time,
            arm_b_finish_time=result_b.timing.finish_time,
            finish_gap_ms=finish_gap_ms,
            signed_finish_error_ms=signed_finish_error_ms,
            arm_a_exec_ms=result_a.timing.exec_ms,
            arm_b_exec_ms=result_b.timing.exec_ms,

            arm_a_target_x=pose_a[0],
            arm_a_target_y=pose_a[1],
            arm_a_target_z=pose_a[2],
            arm_a_target_roll=pose_a[3],
            arm_a_target_pitch=pose_a[4],
            arm_a_target_yaw=pose_a[5],
            arm_b_target_x=pose_b[0],
            arm_b_target_y=pose_b[1],
            arm_b_target_z=pose_b[2],
            arm_b_target_roll=pose_b[3],
            arm_b_target_pitch=pose_b[4],
            arm_b_target_yaw=pose_b[5],

            status="OK",
            error_msg="",
        )
    )

    return record_id + 1


def make_arm_a_pose(
    anchor: Tuple[float, float, float, float, float, float],
    delta: Tuple[float, float, float],
) -> Tuple[float, float, float, float, float, float]:
    ax, ay, az, roll, pitch, yaw = anchor
    dx, dy, dz = delta
    return ax + dx, ay + dy, az + dz, roll, pitch, yaw


def make_arm_b_pose(
    anchor: Tuple[float, float, float, float, float, float],
    delta: Tuple[float, float, float],
) -> Tuple[float, float, float, float, float, float]:
    bx, by, bz, roll, pitch, yaw = anchor
    dx, dy, dz = delta
    return (
        bx + ARM_B_DX_SIGN * dx,
        by + ARM_B_DY_SIGN * dy,
        bz + ARM_B_DZ_SIGN * dz,
        roll,
        pitch,
        yaw,
    )


def print_status(arm: RobotArm) -> None:
    print(f"\n[{arm.name}] Status Check:")
    print(f"  Connected : {arm.arm.connected}")
    print(f"  State     : {arm.arm.state}")
    print(f"  Error Code: {arm.arm.error_code}")
    print(f"  Warn Code : {arm.arm.warn_code}")
    print(f"  Joint Angles: {arm.get_joints()}")
    print(f"  TCP Pose    : {arm.get_pose()}")


def move_to_safe_ready_poses(arm_a: RobotArm, arm_b: RobotArm) -> None:
    print("\nMoving both arms to safe ready joint poses...")
    arm_a.move_joints_retry(READY_JOINTS_A)
    arm_b.move_joints_retry(READY_JOINTS_B)
    time.sleep(1.0)


def main() -> None:
    arm_a = RobotArm(ARM_A_IP, "Arm-A")
    arm_b = RobotArm(ARM_B_IP, "Arm-B")
    log_entries: List[SyncLogEntry] = []
    record_id = 1

    disturbance_plan = build_disturbance_plan(NUM_CYCLES)
    save_disturbance_plan(
        disturbance_plan,
        DISTURBANCE_PLAN_CSV_PATH,
    )

    disturbed_count = sum(
        1 for spec in disturbance_plan.values() if spec.enabled
    )
    total_count = NUM_CYCLES * len(TASK_PATTERN)

    print(
        f"[PLAN] {disturbed_count}/{total_count} phase executions "
        f"({100.0 * disturbed_count / total_count:.1f}%) contain a disturbance."
    )
    print(f"[PLAN] Saved to: {DISTURBANCE_PLAN_CSV_PATH}")

    try:
        arm_a.connect()
        arm_b.connect()
        print_status(arm_a)
        print_status(arm_b)

        move_to_safe_ready_poses(arm_a, arm_b)

        anchor_a = tuple(arm_a.get_pose())
        anchor_b = tuple(arm_b.get_pose())

        print(f"\n[ANCHOR] Arm-A: {[f'{v:.2f}' for v in anchor_a]}")
        print(f"[ANCHOR] Arm-B: {[f'{v:.2f}' for v in anchor_b]}")

        total_steps = NUM_CYCLES * len(TASK_PATTERN)

        for step in range(total_steps):
            phase_name, delta = TASK_PATTERN[step % len(TASK_PATTERN)]
            cycle_index = step // len(TASK_PATTERN) + 1
            phase_index = step % len(TASK_PATTERN) + 1

            pose_a = make_arm_a_pose(anchor_a, delta)
            pose_b = make_arm_b_pose(anchor_b, delta)
            disturbance = disturbance_plan[(cycle_index, phase_index)]

            record_id = synchronized_cartesian_move(
                arm_a=arm_a,
                pose_a=pose_a,
                arm_b=arm_b,
                pose_b=pose_b,
                disturbance=disturbance,
                label=(
                    f"Cycle {cycle_index} / Phase {phase_index}: "
                    f"{phase_name}"
                ),
                cycle_index=cycle_index,
                phase_index=phase_index,
                phase_name=phase_name,
                record_id=record_id,
                log_entries=log_entries,
            )

            time.sleep(PHASE_HOLD_SEC)

        # Return to anchor without disturbance.
        record_id = synchronized_cartesian_move(
            arm_a=arm_a,
            pose_a=anchor_a,
            arm_b=arm_b,
            pose_b=anchor_b,
            disturbance=DisturbanceSpec(False, "", 0.0, 0.0, 1.0),
            label="Return both arms to synchronized anchor poses",
            cycle_index=9999,
            phase_index=0,
            phase_name="ReturnAnchor",
            record_id=record_id,
            log_entries=log_entries,
        )

        save_csv(log_entries, CSV_PATH)
        print(f"\nSaved experiment log to: {CSV_PATH}")
        print(f"{EXPERIMENT_LABEL} disturbance experiment completed.")

    finally:
        print("\nStopping and disconnecting...")
        arm_a.disconnect()
        arm_b.disconnect()


if __name__ == "__main__":
    main()
