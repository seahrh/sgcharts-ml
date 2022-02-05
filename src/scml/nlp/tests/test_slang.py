from scml.nlp import SlangExpansion


class TestSlang:
    def test_no_replacement(self):
        f = SlangExpansion()
        assert f.apply("") == ""
        assert f.apply("bar") == "bar"

    def test_keep_original_term(self):
        f = SlangExpansion(prefix="[", suffix="]", keep_original_term=True)
        assert f.apply("lol") == "[lol; laughing out loud]"

    def test_pattern_is_not_case_sensitive(self):
        f = SlangExpansion(prefix="[", suffix="]", keep_original_term=False)
        assert f.apply("LOL") == "[laughing out loud]"

    def test_replacement_without_punctuation(self):
        f = SlangExpansion(prefix="[", suffix="]", keep_original_term=False)
        assert f.apply("imo") == "[in my opinion]"
        assert f.apply("lol") == "[laughing out loud]"
        assert f.apply("1 lol 2") == "1 [laughing out loud] 2"
        assert f.apply("lol 2") == "[laughing out loud] 2"
        assert f.apply("1 lol") == "1 [laughing out loud]"
        assert f.apply("s4s") == "[share for share; mutual promotion on social media]"
        assert (
            f.apply("1 s4s 2")
            == "1 [share for share; mutual promotion on social media] 2"
        )
        assert (
            f.apply("s4s 2") == "[share for share; mutual promotion on social media] 2"
        )
        assert (
            f.apply("1 s4s") == "1 [share for share; mutual promotion on social media]"
        )

    def test_replacement_with_punctuation(self):
        f = SlangExpansion(prefix="[", suffix="]", keep_original_term=False)
        assert f.apply("+1") == "[expression of agreement or approval]"
        assert f.apply("1 +1 2") == "1 [expression of agreement or approval] 2"
        assert f.apply("+1 2") == "[expression of agreement or approval] 2"
        assert f.apply("1 +1") == "1 [expression of agreement or approval]"
        assert f.apply("v-bag") == "[a repulsive sexual act]"
        assert f.apply("1 v-bag 2") == "1 [a repulsive sexual act] 2"
        assert f.apply("v-bag 2") == "[a repulsive sexual act] 2"
        assert f.apply("1 v-bag") == "1 [a repulsive sexual act]"
        assert f.apply("punk'd") == "[to be the victim of a prank]"
        assert f.apply("1 punk'd 2") == "1 [to be the victim of a prank] 2"
        assert f.apply("punk'd 2") == "[to be the victim of a prank] 2"
        assert f.apply("1 punk'd") == "1 [to be the victim of a prank]"
        assert (
            f.apply("5-0") == "[police officers or warning that police is approaching]"
        )
        assert (
            f.apply("1 5-0 2")
            == "1 [police officers or warning that police is approaching] 2"
        )
        assert (
            f.apply("5-0 2")
            == "[police officers or warning that police is approaching] 2"
        )
        assert (
            f.apply("1 5-0")
            == "1 [police officers or warning that police is approaching]"
        )
        assert (
            f.apply("8-ball")
            == "[1/8 of an ounce (approximately 3.5 grams) of abused drugs]"
        )
        assert (
            f.apply("1 8-ball 2")
            == "1 [1/8 of an ounce (approximately 3.5 grams) of abused drugs] 2"
        )
        assert (
            f.apply("1 8-ball")
            == "1 [1/8 of an ounce (approximately 3.5 grams) of abused drugs]"
        )
        assert (
            f.apply("8-ball 2")
            == "[1/8 of an ounce (approximately 3.5 grams) of abused drugs] 2"
        )
        assert f.apply("b.o.") == "[body odour]"
        assert f.apply("1 b.o. 2") == "1 [body odour] 2"
        assert f.apply("b.o. 2") == "[body odour] 2"
        assert f.apply("1 b.o.") == "1 [body odour]"
