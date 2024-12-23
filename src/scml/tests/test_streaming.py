import numpy as np

from scml import RollingWindow, RunningMean


class TestRollingWindow:
    def test_case_1(self):
        rw = RollingWindow(capacity=4, initial_value=0)
        assert rw.mean() is None
        rw.append(3)
        assert rw.mean() == 3
        rw.append(0)
        assert rw.mean() == 1.5
        rw.append(-3)
        assert rw.mean() == 0
        rw.append(4)
        assert rw.mean() == 1
        # 1st item is popped
        rw.append(7)
        assert rw.mean() == 2
        # 2nd item is popped
        rw.append(2)
        assert rw.mean() == 2.5

    def test_mean(self):
        k = 10
        rw = RollingWindow(capacity=k, initial_value=0)
        assert rw.mean() is None
        ar = np.random.uniform(low=-1000, high=1000, size=(1000,))
        for j in range(ar.shape[0]):
            rw.append(ar[j].item())
            i = max(0, j - k + 1)
            assert np.allclose(rw.mean(), np.mean(ar[i : j + 1]))

    def test_init_buffer(self):
        k = 3
        rw = RollingWindow(capacity=k, initial_value=k, buf=[1] * k)
        assert rw.mean() == 1
        rw.append(4)  # buf = [1, 1, 4]
        assert rw.mean() == 2
        rw.append(4)  # buf = [1, 4, 4]
        assert rw.mean() == 3
        rw.append(7)  # buf = [4, 4, 7]
        assert rw.mean() == 5


class TestRunningMean:

    def test_all_inputs_are_the_same(self):
        rm = RunningMean()
        assert rm.get() == 0
        for _ in range(100):
            rm.add(1)
            assert rm.get() == 1

    def test_case_1(self):
        rm = RunningMean()
        assert rm.get() == 0
        rm.add(3)
        assert rm.get() == 3
        rm.add(0)
        assert rm.get() == 1.5
        rm.add(-3)
        assert rm.get() == 0
        rm.add(4)
        assert rm.get() == 1
        rm.add(6)
        assert rm.get() == 2
        rm.add(2)
        assert rm.get() == 2

    def test_mean(self):
        rm = RunningMean()
        ar = np.random.uniform(low=-1000, high=1000, size=(1000,))
        for i in range(ar.shape[0]):
            rm.add(ar[i].item())
            assert np.allclose(rm.get(), np.mean(ar[: i + 1]))
