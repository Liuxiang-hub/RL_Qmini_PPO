from .BIRL import BIRL


class BIRLStage1(BIRL):
    """Stage 1: remove the reward floor and constrain roll/pitch rates."""

    class posture_reward(BIRL.posture_reward):
        mode = 'posture_rate'
