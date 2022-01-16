from scml.nlp import Slang


class TestSlang:
    def test_no_replacement(self):
        slang = Slang()
        assert slang.expand("") == ""
        assert slang.expand("bar") == "bar"

    def test_replacement_without_punctuation(self):
        slang = Slang(prefix="[", suffix="]")
        assert slang.expand("1 lol 2") == "1 [laughing out loud] 2"
        assert slang.expand("lol 2") == "[laughing out loud] 2"
        assert slang.expand("1 lol") == "1 [laughing out loud]"
        assert (
            slang.expand("1 s4s 2")
            == "1 [share for share; mutual promotion on social media] 2"
        )
        assert (
            slang.expand("s4s 2")
            == "[share for share; mutual promotion on social media] 2"
        )
        assert (
            slang.expand("1 s4s")
            == "1 [share for share; mutual promotion on social media]"
        )

    def test_replacement_with_punctuation(self):
        slang = Slang(prefix="[", suffix="]")
        assert slang.expand("1 +1 2") == "1 [expression of agreement or approval] 2"
        assert slang.expand("+1 2") == "[expression of agreement or approval] 2"
        assert slang.expand("1 +1") == "1 [expression of agreement or approval]"
        assert slang.expand("1 v-bag 2") == "1 [a repulsive sexual act] 2"
        assert slang.expand("v-bag 2") == "[a repulsive sexual act] 2"
        assert slang.expand("1 v-bag") == "1 [a repulsive sexual act]"
        assert slang.expand("1 punk'd 2") == "1 [to be the victim of a prank] 2"
        assert slang.expand("punk'd 2") == "[to be the victim of a prank] 2"
        assert slang.expand("1 punk'd") == "1 [to be the victim of a prank]"
        assert (
            slang.expand("1 5-0 2")
            == "1 [police officers or warning that police is approaching] 2"
        )
        assert (
            slang.expand("5-0 2")
            == "[police officers or warning that police is approaching] 2"
        )
        assert (
            slang.expand("1 5-0")
            == "1 [police officers or warning that police is approaching]"
        )
        assert (
            slang.expand("1 8-ball 2")
            == "1 [1/8 of an ounce (approximately 3.5 grams) of abused drugs] 2"
        )
        assert (
            slang.expand("1 8-ball")
            == "1 [1/8 of an ounce (approximately 3.5 grams) of abused drugs]"
        )
        assert (
            slang.expand("8-ball 2")
            == "[1/8 of an ounce (approximately 3.5 grams) of abused drugs] 2"
        )
        assert slang.expand("1 b.o. 2") == "1 [body odour] 2"
        assert slang.expand("b.o. 2") == "[body odour] 2"
        assert slang.expand("1 b.o.") == "1 [body odour]"
