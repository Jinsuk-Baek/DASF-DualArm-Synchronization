import csv
import threading
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

from xarm import XArmAPI

ARM_A_IP = "192.168.0.251"
ARM_B_IP = "192.168.0.252"

# -----------------------------
# Global motion parameters
# -----------------------------
BASE_CART_SPEED = 80      # mm/s
CART_ACC = 800            # mm/s^2
JOINT_SPEED = 20          # deg/s
JOINT_ACC = 200           # deg/s^2
WAIT = True

# -----------------------------
# TLS parameters
# -----------------------------
BARRIER_TIMEOUT_SEC = 5.0
THREAD_JOIN_TIMEOUT_SEC = 25.0

# -----------------------------
# Stable execution-level MLS parameters
# -----------------------------
MLS_ENABLED = True
MLS_ALPHA = 0.20              # EMA smoothing
MLS_GAIN = 0.002              # very small gain for stable adaptation
MLS_MIN_SCALE = 0.95
MLS_MAX_SCALE = 1.05
MLS_MIN_SAMPLES = 1
MLS_LEARN_ON_RETURN = False
MLS_APPLY_ON_RETURN = False

# -----------------------------
# Demo timing
# -----------------------------
DEMO_DURATION_SEC = 90.0
PHASE_HOLD_SEC = 0.20

# -----------------------------
# CSV logging
# -----------------------------
CSV_PATH = "dual_arm_tls_mls_execspeed_v5_log.csv"

# -----------------------------
# Safe ready joint poses
# -----------------------------
READY_JOINTS_A = [5.0, -20.0, 50.0, 0.0, 10.0, 0.0]
READY_JOINTS_B = [-10.0, -20.0, 50.0, 0.0, 15.0, 0.0]

# -----------------------------
# Axis mapping for Arm-B
# -----------------------------
ARM_B_DX_SIGN = 1.0
ARM_B_DY_SIGN = 1.0
ARM_B_DZ_SIGN = 1.0

# -----------------------------
# Four-phase task-like Cartesian cycle
# -----------------------------
TASK_PATTERN: List[Tuple[str, Tuple[float, float, float]]] = [
    ("Approach",      (0.0,   0.0, -30.0)),
    ("Lift",          (0.0,   0.0,  30.0)),
    ("Transfer-1",    (70.0,  0.0,  25.0)),
    ("Transfer-2",    (95.0, 55.0,  15.0)),
    ("Place",         (60.0, 85.0,   0.0)),
    ("Retreat",       (20.0, 45.0,  25.0)),
    ("Return-1",      (-30.0, 10.0, 20.0)),
    ("Return-2",      (0.0,   0.0,   0.0)),
]

# Phase-specific speed scale estimates
phase_scale_a: Dict[str, float] = {}
phase_scale_b: Dict[str, float] = {}
phase_sample_count: Dict[str, int] = {}


def now() -> float:
    return time.perf_counter()


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
    mls_delay_ms: float
    arm_a_dispatch_time: float
    arm_b_dispatch_time: float
    dispatch_gap_ms: float
    arm_a_finish_time: float
    arm_b_finish_time: float
    finish_gap_ms: float
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

    def ensure_ready(self) -> None:
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
            f"(state={self.arm.state}, error={self.arm.error_code}, warn={self.arm.warn_code})"
        )

    def connect(self) -> None:
        print(f"[{self.name}] Connecting to {self.ip}...")
        if not self.arm.connected:
            raise RuntimeError(f"[{self.name}] Connection failed")
        self.ensure_ready()
        print(f"[{self.name}] Connected successfully")

    def get_pose(self) -> List[float]:
        ret = self.arm.get_position(is_radian=False)
        if ret[0] != 0:
            raise RuntimeError(f"[{self.name}] Failed to read TCP pose, ret={ret[0]}")
        pose = list(ret[1])
        if len(pose) < 6:
            raise RuntimeError(f"[{self.name}] Invalid TCP pose: {pose}")
        return pose[:6]

    def get_joints(self):
        return self.arm.get_servo_angle(is_radian=False)

    def move_pose(
        self,
        pose: Tuple[float, float, float, float, float, float],
        speed: float,
    ) -> int:
        self.ensure_ready()
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
            wait=WAIT,
            is_radian=False,
        )
        return ret[0] if isinstance(ret, tuple) else ret

    def move_pose_retry(
        self,
        pose: Tuple[float, float, float, float, float, float],
        speed: float,
        retries: int = 2,
    ) -> int:
        last_ret = None
        last_err = None
        for _ in range(retries + 1):
            try:
                ret = self.move_pose(pose, speed=speed)
                if ret == 0:
                    return ret
                last_ret = ret
            except Exception as e:
                last_err = str(e)
            time.sleep(0.3)
            self.ensure_ready()
        if last_err is not None:
            raise RuntimeError(f"[{self.name}] Cartesian move retry failed: {last_err}")
        raise RuntimeError(f"[{self.name}] Cartesian move retry failed, ret={last_ret}")

    def move_joints(self, joints: List[float]) -> int:
        self.ensure_ready()
        ret = self.arm.set_servo_angle(
            angle=joints,
            speed=JOINT_SPEED,
            mvacc=JOINT_ACC,
            wait=WAIT,
            is_radian=False,
        )
        return ret[0] if isinstance(ret, tuple) else ret

    def move_joints_retry(self, joints: List[float], retries: int = 2) -> int:
        last_ret = None
        last_err = None
        for _ in range(retries + 1):
            try:
                ret = self.move_joints(joints)
                if ret == 0:
                    return ret
                last_ret = ret
            except Exception as e:
                last_err = str(e)
            time.sleep(0.3)
            self.ensure_ready()
        if last_err is not None:
            raise RuntimeError(f"[{self.name}] Joint move retry failed: {last_err}")
        raise RuntimeError(f"[{self.name}] Joint move retry failed, ret={last_ret}")

    def disconnect(self) -> None:
        try:
            self.arm.disconnect()
        except Exception:
            pass


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


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def is_return_like_phase(phase_name: str) -> bool:
    return phase_name == "Anchor" or phase_name == "ReturnAnchor" or phase_name.startswith("Return")


def should_apply_mls(phase_name: str) -> bool:
    if not MLS_ENABLED:
        return False
    if not MLS_APPLY_ON_RETURN and is_return_like_phase(phase_name):
        return False
    return phase_sample_count.get(phase_name, 0) >= MLS_MIN_SAMPLES


def should_learn_mls(phase_name: str) -> bool:
    if not MLS_ENABLED:
        return False
    if not MLS_LEARN_ON_RETURN and is_return_like_phase(phase_name):
        return False
    return True


def get_phase_scales(phase_name: str) -> Tuple[float, float]:
    return phase_scale_a.get(phase_name, 1.0), phase_scale_b.get(phase_name, 1.0)


def update_phase_speed_scales(phase_name: str, exec_a_ms: float, exec_b_ms: float) -> Tuple[float, float]:
    """
    Stable one-sided MLS:
    - Only speed up the slower arm slightly.
    - Do not slow down the faster arm.
    """
    current_a = phase_scale_a.get(phase_name, 1.0)
    current_b = phase_scale_b.get(phase_name, 1.0)

    diff_ms = exec_a_ms - exec_b_ms

    target_a = current_a
    target_b = current_b

    if diff_ms > 0:
        # Arm-A slower -> slightly speed up Arm-A only
        target_a = current_a + MLS_GAIN * diff_ms
    elif diff_ms < 0:
        # Arm-B slower -> slightly speed up Arm-B only
        target_b = current_b + MLS_GAIN * (-diff_ms)

    updated_a = (1.0 - MLS_ALPHA) * current_a + MLS_ALPHA * target_a
    updated_b = (1.0 - MLS_ALPHA) * current_b + MLS_ALPHA * target_b

    updated_a = clamp(updated_a, MLS_MIN_SCALE, MLS_MAX_SCALE)
    updated_b = clamp(updated_b, MLS_MIN_SCALE, MLS_MAX_SCALE)

    phase_scale_a[phase_name] = updated_a
    phase_scale_b[phase_name] = updated_b
    phase_sample_count[phase_name] = phase_sample_count.get(phase_name, 0) + 1

    return updated_a, updated_b


def run_arm_motion(
    arm: RobotArm,
    pose: Tuple[float, float, float, float, float, float],
    barrier: threading.Barrier,
    output: Dict[str, ThreadResult],
    key: str,
    speed: float,
) -> None:
    try:
        barrier.wait(timeout=BARRIER_TIMEOUT_SEC)

        dispatch_t = now()
        ret = arm.move_pose_retry(pose, speed=speed, retries=2)
        finish_t = now()

        if ret != 0:
            raise RuntimeError(f"[{arm.name}] move failed, ret={ret}")

        output[key] = ThreadResult(
            success=True,
            timing=ArmTiming(
                dispatch_time=dispatch_t,
                finish_time=finish_t,
                exec_ms=(finish_t - dispatch_t) * 1000.0,
            ),
            ret_code=ret,
        )

    except threading.BrokenBarrierError:
        output[key] = ThreadResult(success=False, error=f"[{arm.name}] barrier broken or timeout")
    except Exception as e:
        try:
            barrier.abort()
        except Exception:
            pass
        output[key] = ThreadResult(success=False, error=f"[{arm.name}] {str(e)}")


def save_csv(log_entries: List[SyncLogEntry], csv_path: str) -> None:
    if not log_entries:
        return
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(log_entries[0]).keys()))
        writer.writeheader()
        for entry in log_entries:
            writer.writerow(asdict(entry))


def synchronized_cartesian_move(
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
) -> int:
    scale_a, scale_b = get_phase_scales(phase_name)

    if not should_apply_mls(phase_name):
        scale_a = 1.0
        scale_b = 1.0

    speed_a = clamp(BASE_CART_SPEED * scale_a, BASE_CART_SPEED * MLS_MIN_SCALE, BASE_CART_SPEED * MLS_MAX_SCALE)
    speed_b = clamp(BASE_CART_SPEED * scale_b, BASE_CART_SPEED * MLS_MIN_SCALE, BASE_CART_SPEED * MLS_MAX_SCALE)

    print(f"\n[SYNC-TLS+MLS] {label}")
    print(f"[SYNC-TLS+MLS] Arm-A target: {[f'{v:.2f}' for v in pose_a]}")
    print(f"[SYNC-TLS+MLS] Arm-B target: {[f'{v:.2f}' for v in pose_b]}")
    print(
        f"[MLS] Phase='{phase_name}', samples={phase_sample_count.get(phase_name, 0)}, "
        f"apply={'YES' if should_apply_mls(phase_name) else 'NO'}, "
        f"scaleA={scale_a:.3f}, scaleB={scale_b:.3f}, "
        f"speedA={speed_a:.1f}, speedB={speed_b:.1f}"
    )

    barrier = threading.Barrier(2)
    output: Dict[str, ThreadResult] = {}

    th_a = threading.Thread(target=run_arm_motion, args=(arm_a, pose_a, barrier, output, "A", speed_a), daemon=True)
    th_b = threading.Thread(target=run_arm_motion, args=(arm_b, pose_b, barrier, output, "B", speed_b), daemon=True)

    th_a.start()
    th_b.start()
    th_a.join(timeout=THREAD_JOIN_TIMEOUT_SEC)
    th_b.join(timeout=THREAD_JOIN_TIMEOUT_SEC)

    if th_a.is_alive() or th_b.is_alive():
        try:
            barrier.abort()
        except Exception:
            pass
        raise RuntimeError(f"[SYNC-TLS+MLS] Thread join timeout during '{label}'")

    if "A" not in output or "B" not in output:
        raise RuntimeError(f"[SYNC-TLS+MLS] Missing thread output during '{label}'")

    res_a = output["A"]
    res_b = output["B"]

    if not res_a.success or not res_b.success:
        msg_a = res_a.error if not res_a.success else ""
        msg_b = res_b.error if not res_b.success else ""
        raise RuntimeError(f"[SYNC-TLS+MLS] Motion failed during '{label}': {msg_a} {msg_b}".strip())

    assert res_a.timing is not None
    assert res_b.timing is not None

    dispatch_gap_ms = abs(res_a.timing.dispatch_time - res_b.timing.dispatch_time) * 1000.0
    finish_gap_ms = abs(res_a.timing.finish_time - res_b.timing.finish_time) * 1000.0

    print(f"[MLS] Dispatch gap: {dispatch_gap_ms:.2f} ms")
    print(f"[MLS] Finish gap  : {finish_gap_ms:.2f} ms")

    if should_learn_mls(phase_name):
        new_a, new_b = update_phase_speed_scales(phase_name, res_a.timing.exec_ms, res_b.timing.exec_ms)
        print(f"[MLS] Updated scales for '{phase_name}': scaleA={new_a:.3f}, scaleB={new_b:.3f}")
    else:
        print(f"[MLS] Learning skipped for phase '{phase_name}'")

    effective_scale_marker = ((scale_a - 1.0) + (scale_b - 1.0)) * 0.5 * 1000.0

    log_entries.append(
        SyncLogEntry(
            record_id=record_id,
            phase_index=phase_index,
            cycle_index=cycle_index,
            phase_name=phase_name,
            label=label,
            mls_delay_ms=effective_scale_marker,
            arm_a_dispatch_time=res_a.timing.dispatch_time,
            arm_b_dispatch_time=res_b.timing.dispatch_time,
            dispatch_gap_ms=dispatch_gap_ms,
            arm_a_finish_time=res_a.timing.finish_time,
            arm_b_finish_time=res_b.timing.finish_time,
            finish_gap_ms=finish_gap_ms,
            arm_a_exec_ms=res_a.timing.exec_ms,
            arm_b_exec_ms=res_b.timing.exec_ms,
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


def make_arm_a_pose(anchor: Tuple[float, float, float, float, float, float], delta: Tuple[float, float, float]) -> Tuple[float, float, float, float, float, float]:
    ax, ay, az, roll, pitch, yaw = anchor
    dx, dy, dz = delta
    return (ax + dx, ay + dy, az + dz, roll, pitch, yaw)


def make_arm_b_pose(anchor: Tuple[float, float, float, float, float, float], delta: Tuple[float, float, float]) -> Tuple[float, float, float, float, float, float]:
    bx, by, bz, roll, pitch, yaw = anchor
    dx, dy, dz = delta
    return (bx + ARM_B_DX_SIGN * dx, by + ARM_B_DY_SIGN * dy, bz + ARM_B_DZ_SIGN * dz, roll, pitch, yaw)


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
    duration_sec: float,
    log_entries: List[SyncLogEntry],
    record_id: int,
) -> int:
    print(f"\n[DEMO] Running 4-phase task-like TLS+MLS synchronized Cartesian demo for {duration_sec:.1f} seconds...")
    start_t = now()
    step = 0

    while (now() - start_t) < duration_sec:
        phase_name, delta = TASK_PATTERN[step % len(TASK_PATTERN)]
        cycle_index = (step // len(TASK_PATTERN)) + 1
        phase_index = (step % len(TASK_PATTERN)) + 1

        pose_a = make_arm_a_pose(anchor_a, delta)
        pose_b = make_arm_b_pose(anchor_b, delta)

        record_id = synchronized_cartesian_move(
            arm_a, pose_a, arm_b, pose_b,
            label=f"Phase {phase_index}: {phase_name}",
            phase_index=phase_index, cycle_index=cycle_index, phase_name=phase_name,
            record_id=record_id, log_entries=log_entries,
        )

        time.sleep(PHASE_HOLD_SEC)
        step += 1

    print("[DEMO] TLS+MLS task-like Cartesian demo completed.")
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

        print(f"\n[ANCHOR] Arm-A anchor: {[f'{v:.2f}' for v in anchor_a]}")
        print(f"[ANCHOR] Arm-B anchor: {[f'{v:.2f}' for v in anchor_b]}")

        record_id = synchronized_cartesian_move(
            arm_a, anchor_a, arm_b, anchor_b,
            label="Move both arms to synchronized anchor poses",
            phase_index=0, cycle_index=0, phase_name="Anchor",
            record_id=record_id, log_entries=log_entries,
        )

        record_id = run_task_like_demo(
            arm_a, arm_b, anchor_a, anchor_b,
            duration_sec=DEMO_DURATION_SEC,
            log_entries=log_entries,
            record_id=record_id,
        )

        record_id = synchronized_cartesian_move(
            arm_a, anchor_a, arm_b, anchor_b,
            label="Return both arms to synchronized anchor poses",
            phase_index=0, cycle_index=9999, phase_name="ReturnAnchor",
            record_id=record_id, log_entries=log_entries,
        )

        save_csv(log_entries, CSV_PATH)
        print(f"\nSaved CSV log to: {CSV_PATH}")
        print("\n✔ Both arms completed the 4-phase TLS+MLS synchronized demo successfully.")

    finally:
        print("\nDisconnecting...")
        arm_a.disconnect()
        arm_b.disconnect()


if __name__ == "__main__":
    main()
