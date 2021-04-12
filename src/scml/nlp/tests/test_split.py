from scml.nlp import split


class TestSplit:
    def test_delimiter_length_equals_1(self):
        assert split(delimiters=["a"], s="a1a2a",) == ["", "1", "2", ""]
        assert split(delimiters=["a", "b"], s="ab1ba2ab",) == [
            "",
            "",
            "1",
            "",
            "2",
            "",
            "",
        ]

    def test_delimiter_length_greater_than_1(self):
        assert split(
            delimiters=["a", "...", "(c)"],
            s="stackoverflow (c) is awesome... isn't it?",
        ) == ["st", "ckoverflow ", " is ", "wesome", " isn't it?"]

    def test_punctuation(self):
        assert split(
            delimiters=["!", ".", "?", ")", "("],
            s="hi there! greetings. how are you? (foo) end",
        ) == ["hi there", " greetings", " how are you", " ", "foo", " end"]
