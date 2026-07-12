import numpy as np
from math import pi

from .Base import SetDict2Class, Base


class BIRL(Base):
    def __init__(self):
        super(BIRL, self).__init__

    class task(SetDict2Class):
        cfg = 'BIRL'

    class action(SetDict2Class):
        action_limit_up = None
        action_limit_low = None

        high_ranges = [3.] * 2 + [1.] * 10
        low_ranges = [0.5] * 2 + [-1.] * 10

        ref_joint_pos = [0.4, -0.1, -1.5, 1., -1.3,  -0.4, 0.1, 1.5, -1., 1.3]

        use_increment = True
        inc_high_ranges = [3.5] * 2 + [15.] * 10
        inc_low_ranges = [0.5] * 2 + [-15.] * 10
