from scml.nlp import Slang


class TestSlang:
    def test_no_replacement(self):
        slang = Slang()
        assert slang.expand("") == ""
        assert slang.expand("bar") == "bar"

    def test_replacement_without_punctuation(self):
        slang = Slang(prefix="[", suffix="]")
        assert slang.expand("1 asap 2") == "1 [as soon as possible] 2"

    def test_replacement_with_punctuation(self):
        slang = Slang(prefix="[", suffix="]")
        assert slang.expand("1 +1 2") == "1 [I agree] 2"
        assert slang.expand("1 punk'd 2") == "1 [to be the victim of a prank] 2"
        assert (
            slang.expand("1 5-0 2")
            == "1 [police officers or warning that police is approaching] 2"
        )
        assert (
            slang.expand("1 8-ball 2")
            == "1 [1/8 of an ounce (approximately 3.5 grams) of abused drugs] 2"
        )
        assert slang.expand("1 b.o. 2") == "1 [body odour] 2"
