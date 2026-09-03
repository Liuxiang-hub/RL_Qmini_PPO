import numpy as np
from math import pi

from .Base import SetDict2Class, Base


class BIRL(Base):
    def __init__(self):
        super(BIRL, self).__init__

    class task(SetDict2Class):
        cfg = 'BIRL'

    class posture_reward(SetDict2Class):
        """Posture constraint used by BIRLTask.reward().

        Modes:
          baseline   Original RoboTamer reward, including the 0.5 reward floor.
          dynamic_reference Original reward evaluated around a state-dependent
            roll/pitch reference instead of an always-upright reference.
          angle_only Height and roll/pitch error without the reward floor.
          posture_rate Angle and roll/pitch rate constraint without the floor.
        """
        # Keep the historical `--config BIRL` command backward compatible.
        mode = 'baseline'

        target_height = 0.45
        height_gain = 70.0

        angle_gain = 8.0
        rate_gain = 1.5

        # Dynamic-reference gains and safety limits (radians).
        lateral_velocity_gain = 0.12
        forward_velocity_error_gain = 0.10
        roll_rate_gain = 0.03
        pitch_rate_gain = 0.03
        max_roll_reference = np.deg2rad(8.0)
        max_pitch_reference = np.deg2rad(6.0)

    class action(SetDict2Class):
        action_limit_up = None
        action_limit_low = None

        high_ranges = [3.] * 2 + [1.] * 10
        low_ranges = [0.5] * 2 + [-1.] * 10

        ref_joint_pos = [0.55, 0.25, -1.35, 1.2, -1.1, -0.55, -0.25, 1.35, -1.2, 1.1]

        use_increment = True
        inc_high_ranges = [3.5] * 2 + [15.] * 10
        inc_low_ranges = [0.5] * 2 + [-15.] * 10
