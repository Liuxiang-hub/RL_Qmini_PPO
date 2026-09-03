from .BIRL import BIRL


class BIRLAngleOnly(BIRL):
    """Paper ablation B: posture angle constraint without a reward floor."""

    class posture_reward(BIRL.posture_reward):
        mode = 'angle_only'
