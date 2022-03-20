import math
import time

from scml.timex import *


class TestTimer:
    def test_measure_elapsed_time_inside_context(self):
        with Timer() as tim:
            time.sleep(0.5)
        time.sleep(0.5)
        assert math.isclose(0.5, tim.elapsed.total_seconds(), abs_tol=1e-2)
