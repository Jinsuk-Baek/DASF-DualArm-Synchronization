import csv
import math
import threading
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

from xarm import XArmAPI


ARM_A_IP = "192.168.0.251"
ARM_B_IP = "192.168.0.252"

# ============================================================
# Global motion parameters
# ============================================================
BASE_CART_SPEED = 80.0       # mm/s
CART_ACC = 800.0             # mm/s^2
JOINT_SPEED = 20.0           # deg/s
JOINT_ACC = 200.0            # deg/s^2

# ============================================================
# Synchronization / thread parameters
# ============================================================
BARRIER_TIMEOUT_SEC = 5.0
THREAD_JOIN_TIMEOUT_SEC = 30.0

# ============================================================
# Velocity-mode RT-MLS parameters
# ============================================================
CONTROL_INTERVAL_SEC = 0.05  # 20 Hz velocity-command update
VELOCITY_COMMAND_DURATION_SEC = 0.10

# Translation-only target tolerance. All task poses keep the anchor orientation.
POSITION_TOLERANCE_MM = 1.5
FINAL_CONFIRM_SAMPLES = 2

# Slow down near the target to reduce overshoot.
APPROACH_ZONE_MM = 25.0
MIN_APPROACH_SPEED_MM_S = 8.0

# RT-MLS uses normalized path progress:
# progress = 1 - remaining_distance / initial_distance
#
# Positive progress_gap means Arm-A is ahead of Arm-B.
MLS_PROGRESS_DEADBAND = 0.010        # 1% normalized progress
MLS_KP = 0.35                        # scale correction per unit progress gap
MLS_MAX_STEP_CORRECTION = 0.025      # maximum scale change per 50-ms update

# Absolute commanded-speed scale bounds relative to BASE_CART_SPEED.
# Arm-B starts at 0.80, but RT-MLS is allowed to raise it above 1.0.
MLS_MIN_SCALE = 0.65
MLS_MAX_SCALE = 1.15

# Small relaxation toward each arm's initial scale when synchronized.
MLS_RELAXATION = 0.02

# ============================================================
# Controlled asymmetric workload
# ============================================================
ARM_B_HANDICAP_SCALE = 0.80

# ============================================================
# Experiment parameters
# ============================================================
NUM_CYCLES = 30
PHASE_HOLD_SEC = 0.20

CSV_PATH = "dual_arm_tls_mls_rt_velocity_asym_workload_log.csv"

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
    ret_code: Optional[int] = None


@dataclass
class SyncLogEntry:
    record_id: int
    phase_index: int
    cycle_index: int
    phase_name: str
    label: str

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
            f"[{self.name}] Robot is not ready "
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
                f"[{self.name}] Error after entering velocity mode: "
                f"{self.arm.error_code}"
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
            raise RuntimeError(f"[{self.name}] Failed to read TCP pose: {ret}")
        pose = list(ret[1])
        if len(pose) < 6:
            raise RuntimeError(f"[{self.name}] Invalid TCP pose: {pose}")
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
                f"[{self.name}] Failed to stop Cartesian velocity, ret={ret}"
            )

    def move_pose_blocking(
        self,
        pose: Tuple[float, float, float, float, float, float],
        speed: float = BASE_CART_SPEED,
    ) -> int:
        self.ensure_ready_position_mode()
        x, y, z, roll, pitch, yaw = pose
        ret = self.arm.set_position(
            x=x,
            y=y,
            z=z,
            roll=roll,
            pitch=pitch,
            yaw=yaw,
            speed=speed,
            mvacc=CART_ACC,
            wait=True,
            is_radian=False,
        )
        return self._ret_code(ret)

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
            raise RuntimeError(f"[{self.name}] Joint move failed: {last_error}")
        raise RuntimeError(f"[{self.name}] Joint move failed, ret={last_ret}")

    def disconnect(self) -> None:
        try:
            self.exit_cartesian_velocity_mode()
        except Exception:
            pass
        try:
            self.arm.disconnect()
        except Exception:
            pass


class RTMLSState:
    def __init__(self, init_scale_a: float, init_scale_b: float) -> None:
        self.lock = threading.Lock()
        self.stop_event = threading.Event()

        self.init_scale_a = init_scale_a
        self.init_scale_b = init_scale_b

        # These are the actual absolute velocity-command scales relative
        # to BASE_CART_SPEED, not unused bookkeeping variables.
        self.scale_a = init_scale_a
        self.scale_b = init_scale_b

        self.progress_a = 0.0
        self.progress_b = 0.0
        self.remaining_a = 0.0
        self.remaining_b = 0.0
        self.total_distance_a = 0.0
        self.total_distance_b = 0.0

        self.start_a = 0.0
        self.start_b = 0.0
        self.finish_a = 0.0
        self.finish_b = 0.0
        self.a_done = False
        self.b_done = False

        self.scale_history_a: List[float] = [init_scale_a]
        self.scale_history_b: List[float] = [init_scale_b]
        self.update_count = 0
        self.max_abs_progress_gap = 0.0


def rt_mls_progress_controller(state: RTMLSState) -> None:
    """
    Closed-loop RT-MLS controller.

    The controller compares normalized path progress, not elapsed time.
    If Arm-A is ahead, Arm-A is slowed and Arm-B is accelerated.
    If Arm-B is ahead, the opposite correction is applied.

    The updated scales are read by the velocity-command loops and therefore
    directly alter the commands sent through vc_set_cartesian_velocity().
    """
    while not state.stop_event.wait(CONTROL_INTERVAL_SEC):
        with state.lock:
            if state.a_done and state.b_done:
                break

            # Wait until both motion threads have initialized their paths.
            if (
                state.start_a == 0.0
                or state.start_b == 0.0
                or state.total_distance_a <= 0.0
                or state.total_distance_b <= 0.0
            ):
                continue

            progress_gap = state.progress_a - state.progress_b
            state.max_abs_progress_gap = max(
                state.max_abs_progress_gap, abs(progress_gap)
            )

            if state.a_done or state.b_done:
                # Once one arm reaches the target, do not generate a large
                # last-instant correction. The unfinished arm retains its
                # latest valid command scale.
                state.scale_history_a.append(state.scale_a)
                state.scale_history_b.append(state.scale_b)
                continue

            if abs(progress_gap) > MLS_PROGRESS_DEADBAND:
                correction = clamp(
                    MLS_KP * progress_gap,
                    -MLS_MAX_STEP_CORRECTION,
                    MLS_MAX_STEP_CORRECTION,
                )

                # Positive gap: A is ahead -> slow A, accelerate B.
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
                # When synchronized, gently relax toward initial settings
                # instead of leaving the controller at a saturated value.
                state.scale_a += MLS_RELAXATION * (
                    state.init_scale_a - state.scale_a
                )
                state.scale_b += MLS_RELAXATION * (
                    state.init_scale_b - state.scale_b
                )

            state.scale_history_a.append(state.scale_a)
            state.scale_history_b.append(state.scale_b)


def run_arm_velocity_motion(
    arm: RobotArm,
    target_pose: Tuple[float, float, float, float, float, float],
    barrier: threading.Barrier,
    output: Dict[str, ThreadResult],
    key: str,
    state: RTMLSState,
) -> None:
    dispatch_t = 0.0

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
                state.remaining_a = total_distance
            else:
                state.total_distance_b = total_distance
                state.remaining_b = total_distance

        barrier.wait(timeout=BARRIER_TIMEOUT_SEC)

        arm.enter_cartesian_velocity_mode()
        dispatch_t = now()

        with state.lock:
            if key == "A":
                state.start_a = dispatch_t
            else:
                state.start_b = dispatch_t

        # A zero-length move is valid for the initial anchor record.
        if total_distance <= POSITION_TOLERANCE_MM:
            arm.stop_cartesian_velocity()
            finish_t = now()

            with state.lock:
                if key == "A":
                    state.progress_a = 1.0
                    state.remaining_a = 0.0
                    state.finish_a = finish_t
                    state.a_done = True
                else:
                    state.progress_b = 1.0
                    state.remaining_b = 0.0
                    state.finish_b = finish_t
                    state.b_done = True

            output[key] = ThreadResult(
                success=True,
                timing=ArmTiming(
                    dispatch_time=dispatch_t,
                    finish_time=finish_t,
                    exec_ms=(finish_t - dispatch_t) * 1000.0,
                ),
                ret_code=0,
            )
            return

        confirm_count = 0
        loop_start = now()
        motion_timeout = THREAD_JOIN_TIMEOUT_SEC - 3.0

        while True:
            if (now() - loop_start) > motion_timeout:
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
                    state.remaining_a = remaining
                    command_scale = state.scale_a
                else:
                    state.progress_b = progress
                    state.remaining_b = remaining
                    command_scale = state.scale_b

            if remaining <= POSITION_TOLERANCE_MM:
                confirm_count += 1
                arm.stop_cartesian_velocity()

                if confirm_count >= FINAL_CONFIRM_SAMPLES:
                    break

                time.sleep(CONTROL_INTERVAL_SEC)
                continue

            confirm_count = 0
            unit_x, unit_y, unit_z = normalize3(error_x, error_y, error_z)

            commanded_speed = BASE_CART_SPEED * command_scale

            # Linear slowdown in the final approach zone.
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

        finish_t = now()
        arm.stop_cartesian_velocity()

        with state.lock:
            if key == "A":
                state.progress_a = 1.0
                state.remaining_a = 0.0
                state.finish_a = finish_t
                state.a_done = True
            else:
                state.progress_b = 1.0
                state.remaining_b = 0.0
                state.finish_b = finish_t
                state.b_done = True

        output[key] = ThreadResult(
            success=True,
            timing=ArmTiming(
                dispatch_time=dispatch_t,
                finish_time=finish_t,
                exec_ms=(finish_t - dispatch_t) * 1000.0,
            ),
            ret_code=0,
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


def synchronized_cartesian_move_rt_velocity(
    arm_a: RobotArm,
    pose_a: Tuple[float, float, float, float, float, float],
    arm_b: RobotArm,
    pose_b: Tuple[float, float, float, float, float, float],
    label: str,
    phase_index: int,
    cycle_index: int,
    phase_name: str,
    record_id: int,
    log_entries: List[SyncLogEntry],
    init_scale_a: float,
    init_scale_b: float,
) -> int:
    print(f"\n[SYNC-RT-MLS-VC] {label}")
    print(
        f"[SYNC-RT-MLS-VC] Initial scale A/B: "
        f"{init_scale_a:.3f} / {init_scale_b:.3f}"
    )

    state = RTMLSState(init_scale_a, init_scale_b)
    barrier = threading.Barrier(2)
    output: Dict[str, ThreadResult] = {}

    controller_thread = threading.Thread(
        target=rt_mls_progress_controller,
        args=(state,),
        daemon=True,
    )
    thread_a = threading.Thread(
        target=run_arm_velocity_motion,
        args=(arm_a, pose_a, barrier, output, "A", state),
        daemon=True,
    )
    thread_b = threading.Thread(
        target=run_arm_velocity_motion,
        args=(arm_b, pose_b, barrier, output, "B", state),
        daemon=True,
    )

    controller_thread.start()
    thread_a.start()
    thread_b.start()

    thread_a.join(timeout=THREAD_JOIN_TIMEOUT_SEC)
    thread_b.join(timeout=THREAD_JOIN_TIMEOUT_SEC)

    state.stop_event.set()
    controller_thread.join(timeout=2.0)

    # Always stop both arms before returning to position mode.
    for arm in (arm_a, arm_b):
        try:
            arm.stop_cartesian_velocity()
        except Exception:
            pass
        arm.exit_cartesian_velocity_mode()

    if thread_a.is_alive() or thread_b.is_alive():
        raise RuntimeError(
            f"[SYNC-RT-MLS-VC] Thread join timeout during '{label}'"
        )

    if "A" not in output or "B" not in output:
        raise RuntimeError(
            f"[SYNC-RT-MLS-VC] Missing thread output during '{label}'"
        )

    result_a = output["A"]
    result_b = output["B"]

    if not result_a.success or not result_b.success:
        error_a = result_a.error if not result_a.success else ""
        error_b = result_b.error if not result_b.success else ""
        raise RuntimeError(
            f"[SYNC-RT-MLS-VC] Motion failed during '{label}': "
            f"{error_a} {error_b}".strip()
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

        final_scale_a = state.scale_a
        final_scale_b = state.scale_b
        update_count = state.update_count
        max_abs_progress_gap = state.max_abs_progress_gap

    mean_scale_a = sum(history_a) / len(history_a)
    mean_scale_b = sum(history_b) / len(history_b)

    print(f"[RT-MLS] Dispatch gap       : {dispatch_gap_ms:.2f} ms")
    print(f"[RT-MLS] Finish gap         : {finish_gap_ms:.2f} ms")
    print(f"[RT-MLS] Signed finish error: {signed_finish_error_ms:.2f} ms")
    print(
        f"[RT-MLS] Scale A init/final : "
        f"{init_scale_a:.3f} / {final_scale_a:.3f}"
    )
    print(
        f"[RT-MLS] Scale B init/final : "
        f"{init_scale_b:.3f} / {final_scale_b:.3f}"
    )
    print(
        f"[RT-MLS] Scale A min/max    : "
        f"{min(history_a):.3f} / {max(history_a):.3f}"
    )
    print(
        f"[RT-MLS] Scale B min/max    : "
        f"{min(history_b):.3f} / {max(history_b):.3f}"
    )
    print(f"[RT-MLS] Correction updates : {update_count}")

    log_entries.append(
        SyncLogEntry(
            record_id=record_id,
            phase_index=phase_index,
            cycle_index=cycle_index,
            phase_name=phase_name,
            label=label,

            arm_a_init_scale=init_scale_a,
            arm_b_init_scale=init_scale_b,

            rt_scale_a_final=final_scale_a,
            rt_scale_b_final=final_scale_b,
            rt_scale_a_min=min(history_a),
            rt_scale_a_max=max(history_a),
            rt_scale_a_mean=mean_scale_a,
            rt_scale_b_min=min(history_b),
            rt_scale_b_max=max(history_b),
            rt_scale_b_mean=mean_scale_b,
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
    pose = arm.get_pose()
    joints = arm.get_joints()
    print(f"\n[{arm.name}] Status Check:")
    print(f"  Connected : {arm.arm.connected}")
    print(f"  State     : {arm.arm.state}")
    print(f"  Error Code: {arm.arm.error_code}")
    print(f"  Warn Code : {arm.arm.warn_code}")
    print(f"  Joint Angles: {joints}")
    print(f"  TCP Pose    : {pose}")


def move_to_safe_ready_poses(arm_a: RobotArm, arm_b: RobotArm) -> None:
    print("\nMoving both arms to safe ready joint poses...")
    arm_a.move_joints_retry(READY_JOINTS_A)
    arm_b.move_joints_retry(READY_JOINTS_B)
    time.sleep(1.0)


def run_task_like_demo(
    arm_a: RobotArm,
    arm_b: RobotArm,
    anchor_a: Tuple[float, float, float, float, float, float],
    anchor_b: Tuple[float, float, float, float, float, float],
    num_cycles: int,
    log_entries: List[SyncLogEntry],
    record_id: int,
) -> int:
    print(
        f"\n[DEMO] Running {num_cycles}-cycle velocity-mode "
        f"TLS+RT-MLS asymmetric-workload demo "
        f"(Arm-B initial scale={ARM_B_HANDICAP_SCALE:.2f})..."
    )

    total_steps = num_cycles * len(TASK_PATTERN)

    for step in range(total_steps):
        phase_name, delta = TASK_PATTERN[step % len(TASK_PATTERN)]
        cycle_index = step // len(TASK_PATTERN) + 1
        phase_index = step % len(TASK_PATTERN) + 1

        pose_a = make_arm_a_pose(anchor_a, delta)
        pose_b = make_arm_b_pose(anchor_b, delta)

        record_id = synchronized_cartesian_move_rt_velocity(
            arm_a=arm_a,
            pose_a=pose_a,
            arm_b=arm_b,
            pose_b=pose_b,
            label=f"Cycle {cycle_index} / Phase {phase_index}: {phase_name}",
            phase_index=phase_index,
            cycle_index=cycle_index,
            phase_name=phase_name,
            record_id=record_id,
            log_entries=log_entries,
            init_scale_a=1.0,
            init_scale_b=ARM_B_HANDICAP_SCALE,
        )

        time.sleep(PHASE_HOLD_SEC)

    print("[DEMO] Velocity-mode TLS+RT-MLS demo completed.")
    return record_id


def main() -> None:
    arm_a = RobotArm(ARM_A_IP, "Arm-A")
    arm_b = RobotArm(ARM_B_IP, "Arm-B")

    log_entries: List[SyncLogEntry] = []
    record_id = 1

    try:
        arm_a.connect()
        arm_b.connect()

        print_status(arm_a)
        print_status(arm_b)

        move_to_safe_ready_poses(arm_a, arm_b)

        anchor_a = tuple(arm_a.get_pose())
        anchor_b = tuple(arm_b.get_pose())

        print(f"\n[ANCHOR] Arm-A: {[f'{value:.2f}' for value in anchor_a]}")
        print(f"[ANCHOR] Arm-B: {[f'{value:.2f}' for value in anchor_b]}")

        # No separate zero-distance anchor record is required. The arms are
        # already at the captured anchors after the safe-ready move.
        record_id = run_task_like_demo(
            arm_a=arm_a,
            arm_b=arm_b,
            anchor_a=anchor_a,
            anchor_b=anchor_b,
            num_cycles=NUM_CYCLES,
            log_entries=log_entries,
            record_id=record_id,
        )

        # Return to the anchors using the same velocity-mode RT controller,
        # but without the asymmetric initial scale.
        record_id = synchronized_cartesian_move_rt_velocity(
            arm_a=arm_a,
            pose_a=anchor_a,
            arm_b=arm_b,
            pose_b=anchor_b,
            label="Return both arms to synchronized anchor poses",
            phase_index=0,
            cycle_index=9999,
            phase_name="ReturnAnchor",
            record_id=record_id,
            log_entries=log_entries,
            init_scale_a=1.0,
            init_scale_b=1.0,
        )

        save_csv(log_entries, CSV_PATH)
        print(f"\nSaved CSV log to: {CSV_PATH}")
        print("\nVelocity-mode TLS+RT-MLS experiment completed successfully.")

    finally:
        print("\nStopping and disconnecting both arms...")
        arm_a.disconnect()
        arm_b.disconnect()


if __name__ == "__main__":
    main()
