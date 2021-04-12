from scml.nlp import is_number


class TestIsNumber:
    def test_case_1(self):
        assert not is_number("")
        assert is_number("1970")
        assert not is_number("2am")
        assert not is_number("a")
        assert not is_number("NaN")
        assert not is_number("1970-12-31")
        assert is_number("1")
        assert is_number("-1")
        assert is_number("1.2")
        assert is_number("-1.2")
        assert is_number("1e3")
        assert is_number("1e-3")
