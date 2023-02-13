import pytest

from scml.nlp.emoji import *


@pytest.fixture(scope="module")
def get_emoji():
    yield Emoji()


class TestStripEmoji:
    def test_no_replacement(self, get_emoji):
        em = get_emoji
        assert em.strip("") == ""
        assert em.strip("joy") == "joy"

    def test_replacement(self, get_emoji):
        em = get_emoji
        assert len(em.entries) >= 4159
        for entry in em.entries:
            assert em.strip(entry.emoji) == "", f"{entry}"


class TestStripEmojiShortcode:
    def test_no_replacement(self, get_emoji):
        em = get_emoji
        assert em.strip_shortcode("") == ""
        assert em.strip_shortcode("joy") == "joy"

    def test_replacement(self, get_emoji):
        em = get_emoji
        assert em.strip_shortcode(":joy:") == ""
        assert em.strip_shortcode("1 :joy: 2") == "1  2"


class TestEmojiShortcodeToText:
    def test_no_matches(self, get_emoji):
        em = get_emoji
        assert em.shortcode_to_text("") == ""
        assert em.shortcode_to_text(":a=1:") == ":a=1:"

    def test_single_match(self, get_emoji):
        em = get_emoji
        assert em.shortcode_to_text("1 :joy: 2") == "1 (joy) 2"
        assert (
            em.shortcode_to_text("1 :person_in_suit_levitating_light_skin_tone: 2")
            == "1 (person in suit levitating light skin tone) 2"
        )
        assert (
            em.shortcode_to_text("1 :person-in-suit-levitating-light-skin-tone: 2")
            == "1 (person in suit levitating light skin tone) 2"
        )
        assert (
            em.shortcode_to_text("1 :person in suit levitating light skin tone: 2")
            == "1 (person in suit levitating light skin tone) 2"
        )

    def test_multiple_matches(self, get_emoji):
        em = get_emoji
        assert em.shortcode_to_text("1 :joy: 2 :foo: 3") == "1 (joy) 2 (foo) 3"
        assert (
            em.shortcode_to_text(
                "1 :person_in_suit_levitating_light_skin_tone: 2 :foo_bar: 3"
            )
            == "1 (person in suit levitating light skin tone) 2 (foo bar) 3"
        )
        assert (
            em.shortcode_to_text(
                "1 :person-in-suit-levitating-light-skin-tone: 2 :foo-bar: 3"
            )
            == "1 (person in suit levitating light skin tone) 2 (foo bar) 3"
        )
        assert (
            em.shortcode_to_text(
                "1 :person in suit levitating light skin tone: 2 :foo bar: 3"
            )
            == "1 (person in suit levitating light skin tone) 2 (foo bar) 3"
        )
