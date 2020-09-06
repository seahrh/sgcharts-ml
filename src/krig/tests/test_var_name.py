from krig import var_name


class TestVarName:
    def test_when_string_contains_uppercase_chars_then_return_lowercase(self):
        assert var_name("Abc") == "abc"

    def test_when_string_contains_punctuation_then_remove_punctuation(self):
        assert var_name("a!@#$%^&*(){}[]-=_+\"'~`|\\b") == "a_b"

    def test_when_string_contains_whitespace_then_replace_with_single_underscore(self):
        assert var_name(" \n\r\t a \n\r\t b \n\r\t ") == "a_b"

    def test_when_string_is_already_legal_variable_name_then_do_nothing(self):
        assert var_name("a_b") == "a_b"
