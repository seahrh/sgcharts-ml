from scml.nlp import EmoticonToText


class TestEmoticonToText:
    def test_no_replacement(self):
        f = EmoticonToText()
        assert f.apply("") == ""
        assert f.apply("bar") == "bar"

    def test_pattern_is_case_sensitive(self):
        f = EmoticonToText()
        assert f.apply("t.t") == "t.t"

    def test_replacement_is_position_invariant(self):
        f = EmoticonToText()
        assert f.apply(":)") == "[Happy face smiley]"
        assert f.apply("1 :) 2") == "1 [Happy face smiley] 2"
        assert f.apply("1 :)") == "1 [Happy face smiley]"
        assert f.apply(":) 2") == "[Happy face smiley] 2"
        assert f.apply(":-)") == "[Happy face smiley]"
        assert f.apply("1 :-) 2") == "1 [Happy face smiley] 2"
        assert f.apply("1 :-)") == "1 [Happy face smiley]"
        assert f.apply(":-) 2") == "[Happy face smiley] 2"
        assert f.apply("T.T") == "[Sad or Crying]"
        assert f.apply("1 T.T 2") == "1 [Sad or Crying] 2"
        assert f.apply("1 T.T") == "1 [Sad or Crying]"
        assert f.apply("T.T 2") == "[Sad or Crying] 2"
        assert f.apply("o_O") == "[Surprised]"
        assert f.apply("1 o_O 2") == "1 [Surprised] 2"
        assert f.apply("1 o_O") == "1 [Surprised]"
        assert f.apply("o_O 2") == "[Surprised] 2"
        assert (
            f.apply("=p")
            == "[Tongue sticking out, cheeky, playful or blowing a raspberry]"
        )
        assert (
            f.apply("1 =p 2")
            == "1 [Tongue sticking out, cheeky, playful or blowing a raspberry] 2"
        )
        assert (
            f.apply("1 =p")
            == "1 [Tongue sticking out, cheeky, playful or blowing a raspberry]"
        )
        assert (
            f.apply("=p 2")
            == "[Tongue sticking out, cheeky, playful or blowing a raspberry] 2"
        )
