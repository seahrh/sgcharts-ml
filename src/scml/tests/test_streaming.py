from scml import RollingWindow, IterativeMean


class TestRollingWindow:
    def test_mean(self):
        rw = RollingWindow(capacity=4)
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


class TestIterativeMean:
    def test_case_1(self):
        im = IterativeMean()
        assert im.get() == 0
        im.add(3)
        assert im.get() == 3
        im.add(0)
        assert im.get() == 1.5
        im.add(-3)
        assert im.get() == 0
        im.add(4)
        assert im.get() == 1
        im.add(6)
        assert im.get() == 2
        im.add(2)
        assert im.get() == 2
