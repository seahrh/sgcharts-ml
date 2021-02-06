from scml import RollingWindow


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
