"""BIRL task with the physical PD gains used by the final Qmini experiment."""

from .BIRL import BIRL


class BIRLPhysicalPD(BIRL):
    class task(BIRL.task):
        # Reuse the BIRL locomotion task; only actuator/model settings change.
        cfg = "BIRL"

    class asset(BIRL.asset):
        file = "assets/q1/urdf/q1_hip_aligned.urdf"

    class pd_gains(BIRL.pd_gains):
        # Symmetric type-level values used by the final training configuration.
        decimation = 15
        stiffness = {
            "hip_yaw": 40.069,
            "hip_roll": 180.310,
            "hip_pitch": 60.103,
            "knee": 39.819,
            "ankle": 34.309,
        }
        damping = {
            "hip_yaw": 2.003,
            "hip_roll": 18.031,
            "hip_pitch": 2.003,
            "knee": 1.991,
            "ankle": 1.716,
        }

    class domain_rand(BIRL.domain_rand):
        # Milliseconds; covers the measured link and larger delay perturbations.
        delay_joint_ranges = [1, 50]
        delay_rate_ranges = [1, 50]
        delay_angle_ranges = [1, 50]
