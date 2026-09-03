from .BIRL import BIRL


class BIRLBaseline(BIRL):
    """Paper ablation A: the unmodified RoboTamer balance reward."""

    class posture_reward(BIRL.posture_reward):
        mode = 'baseline'
