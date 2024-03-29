from scml.nlp.charencoding import *


class TestToAscii:
    def test_alphabet(self):
        assert to_ascii("abcdefghijklmnopqrstuvwxyz") == "abcdefghijklmnopqrstuvwxyz"

    def test_accented_chars(self):
        aa_map = {
            "À": "A",
            "Á": "A",
            "Â": "A",
            "Ã": "A",
            "Ä": "A",
            "Ç": "C",
            "È": "E",
            "É": "E",
            "Ê": "E",
            "Ë": "E",
            "Ì": "I",
            "Í": "I",
            "Î": "I",
            "Ï": "I",
            "Ñ": "N",
            "Ò": "O",
            "Ó": "O",
            "Ô": "O",
            "Õ": "O",
            "Ö": "O",
            "Š": "S",
            "Ú": "U",
            "Û": "U",
            "Ü": "U",
            "Ù": "U",
            "Ý": "Y",
            "Ÿ": "Y",
            "Ž": "Z",
            "à": "a",
            "á": "a",
            "â": "a",
            "ã": "a",
            "ä": "a",
            "ç": "c",
            "è": "e",
            "é": "e",
            "ê": "e",
            "ë": "e",
            "ì": "i",
            "í": "i",
            "î": "i",
            "ï": "i",
            "ñ": "n",
            "ò": "o",
            "ó": "o",
            "ô": "o",
            "õ": "o",
            "ö": "o",
            "š": "s",
            "ù": "u",
            "ú": "u",
            "û": "u",
            "ü": "u",
            "ý": "y",
            "ÿ": "y",
            "ž": "z",
        }
        for k, v in aa_map.items():
            assert to_ascii(k) == v


class TestCp1252ToUtf8:
    def test_no_error(self):
        assert cp1252_to_utf8("foo ÃÅf bar") == "foo ÃÅf bar"
