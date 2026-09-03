from .BIRL import BIRL


class BIRLDynamicReference(BIRL):
    """Ablation B: original reward weights with dynamic posture references."""

    class posture_reward(BIRL.posture_reward):
        mode = 'dynamic_reference'
