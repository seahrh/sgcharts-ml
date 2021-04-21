from scml.nlp import decode_escaped_bytes


class TestDecodeEscapedBytes:
    def test_case_1(self):
        assert (
            decode_escaped_bytes("Nescafe \\xc3\\x89clair Latte 220ml")
            == "Nescafe Éclair Latte 220ml"
        )
        assert (
            decode_escaped_bytes("RATU \\xe2\\x9d\\xa4 MAYCREATE MOISTURIZING SPRAY")
            == "RATU ❤ MAYCREATE MOISTURIZING SPRAY"
        )
        assert (
            decode_escaped_bytes(
                "\xf0\x9f\x87\xb2\xf0\x9f\x87\xa8MG\xf0\x9f\x87\xb2\xf0\x9f\x87\xa8 Fashion"
            )
            == "🇲🇨MG🇲🇨 Fashion"
        )
