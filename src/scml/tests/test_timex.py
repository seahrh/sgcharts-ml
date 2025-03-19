import math
import time
from datetime import datetime

from scml.timex import *


class TestTimer:
    def test_measure_elapsed_time_inside_context(self):
        with Timer() as tim:
            time.sleep(0.5)
        time.sleep(0.5)
        assert math.isclose(0.5, tim.elapsed.total_seconds(), abs_tol=1e-2)


class TestDaysAgo:

    def test_case_1(self):
        reference = datetime(year=2025, month=1, day=2)
        assert days_ago(days=0, reference=reference) == reference
        assert days_ago(days=1, reference=reference) == datetime(
            year=2025, month=1, day=1
        )
        assert days_ago(days=2, reference=reference) == datetime(
            year=2024, month=12, day=31
        )


class TestLastMonth:
    def test_case_1(self):
        assert last_month(reference=datetime(year=2025, month=1, day=2)) == (
            datetime(year=2024, month=12, day=1),
            datetime(year=2024, month=12, day=31),
        )
        assert last_month(reference=datetime(year=2024, month=3, day=31)) == (
            datetime(year=2024, month=2, day=1),
            datetime(year=2024, month=2, day=29),  # 2024 is a leap year
        )
