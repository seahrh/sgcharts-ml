import re
from typing import Any, NamedTuple, Tuple

__all__ = ["expand_contractions"]


class Rule(NamedTuple):
    pattern: Any
    replacement: str


FLAGS = re.IGNORECASE

# Ordering of the rules matters.
# Contractions based on the following references:
# https://stackoverflow.com/a/19794953/519951
# https://en.wikipedia.org/wiki/Wikipedia:List_of_English_contractions

CONTRACTIONS: Tuple[Rule, ...] = tuple(
    [
        Rule(pattern=re.compile(r"ain't", FLAGS), replacement="are not"),
        Rule(pattern=re.compile(r"aren't", FLAGS), replacement="are not"),
        Rule(pattern=re.compile(r"can't've", FLAGS), replacement="cannot have"),
        Rule(pattern=re.compile(r"can't", FLAGS), replacement="cannot"),
        Rule(pattern=re.compile(r"'cause", FLAGS), replacement="because"),
        Rule(pattern=re.compile(r"couldn't've", FLAGS), replacement="could not have"),
        Rule(pattern=re.compile(r"could've", FLAGS), replacement="could have"),
        Rule(pattern=re.compile(r"couldn't", FLAGS), replacement="could not"),
        Rule(pattern=re.compile(r"didn't", FLAGS), replacement="did not"),
        Rule(pattern=re.compile(r"doesn't", FLAGS), replacement="does not"),
        Rule(pattern=re.compile(r"don't", FLAGS), replacement="do not"),
        Rule(pattern=re.compile(r"hadn't've", FLAGS), replacement="had not have"),
        Rule(pattern=re.compile(r"hadn't", FLAGS), replacement="had not"),
        Rule(pattern=re.compile(r"hasn't", FLAGS), replacement="has not"),
        Rule(pattern=re.compile(r"haven't", FLAGS), replacement="have not"),
        Rule(pattern=re.compile(r"he'd", FLAGS), replacement="he would"),
        Rule(pattern=re.compile(r"he'd've", FLAGS), replacement="he would have"),
        Rule(pattern=re.compile(r"he'll", FLAGS), replacement="he will"),
        Rule(pattern=re.compile(r"he'll've", FLAGS), replacement="he will have"),
        Rule(pattern=re.compile(r"he's", FLAGS), replacement="he is"),
        Rule(pattern=re.compile(r"how'd'y", FLAGS), replacement="how do you"),
        Rule(pattern=re.compile(r"how'd", FLAGS), replacement="how did"),
        Rule(pattern=re.compile(r"how'll", FLAGS), replacement="how will"),
        Rule(pattern=re.compile(r"how's", FLAGS), replacement="how is"),
        Rule(pattern=re.compile(r"I'd", FLAGS), replacement="I would"),
        Rule(pattern=re.compile(r"I'd've", FLAGS), replacement="I would have"),
        Rule(pattern=re.compile(r"I'll", FLAGS), replacement="I will"),
        Rule(pattern=re.compile(r"I'll've", FLAGS), replacement="I will have"),
        Rule(pattern=re.compile(r"I'm", FLAGS), replacement="I am"),
        Rule(pattern=re.compile(r"I've", FLAGS), replacement="I have"),
        Rule(pattern=re.compile(r"isn't", FLAGS), replacement="is not"),
        Rule(pattern=re.compile(r"it'd", FLAGS), replacement="it would"),
        Rule(pattern=re.compile(r"it'd've", FLAGS), replacement="it would have"),
        Rule(pattern=re.compile(r"it'll", FLAGS), replacement="it will"),
        Rule(pattern=re.compile(r"it'll've", FLAGS), replacement="it will have"),
        Rule(pattern=re.compile(r"it's", FLAGS), replacement="it is"),
        Rule(pattern=re.compile(r"let's", FLAGS), replacement="let us"),
        Rule(pattern=re.compile(r"ma'am", FLAGS), replacement="madam"),
        Rule(pattern=re.compile(r"mayn't", FLAGS), replacement="may not"),
        Rule(pattern=re.compile(r"mightn't've", FLAGS), replacement="might not have"),
        Rule(pattern=re.compile(r"might've", FLAGS), replacement="might have"),
        Rule(pattern=re.compile(r"mightn't", FLAGS), replacement="might not"),
        Rule(pattern=re.compile(r"mustn't've", FLAGS), replacement="must not have"),
        Rule(pattern=re.compile(r"must've", FLAGS), replacement="must have"),
        Rule(pattern=re.compile(r"mustn't", FLAGS), replacement="must not"),
        Rule(pattern=re.compile(r"needn't've", FLAGS), replacement="need not have"),
        Rule(pattern=re.compile(r"needn't", FLAGS), replacement="need not"),
        Rule(pattern=re.compile(r"o'clock", FLAGS), replacement="of the clock"),
        Rule(pattern=re.compile(r"oughtn't've", FLAGS), replacement="ought not have"),
        Rule(pattern=re.compile(r"oughtn't", FLAGS), replacement="ought not"),
        Rule(pattern=re.compile(r"shan't've", FLAGS), replacement="shall not have"),
        Rule(pattern=re.compile(r"shan't", FLAGS), replacement="shall not"),
        Rule(pattern=re.compile(r"sha'n't", FLAGS), replacement="shall not"),
        Rule(pattern=re.compile(r"she'd've", FLAGS), replacement="she would have"),
        Rule(pattern=re.compile(r"she'd", FLAGS), replacement="she would"),
        Rule(pattern=re.compile(r"she'll've", FLAGS), replacement="she will have"),
        Rule(pattern=re.compile(r"she'll", FLAGS), replacement="she will"),
        Rule(pattern=re.compile(r"she's", FLAGS), replacement="she is"),
        Rule(pattern=re.compile(r"shouldn't've", FLAGS), replacement="should not have"),
        Rule(pattern=re.compile(r"should've", FLAGS), replacement="should have"),
        Rule(pattern=re.compile(r"shouldn't", FLAGS), replacement="should not"),
        Rule(pattern=re.compile(r"so've", FLAGS), replacement="so have"),
        Rule(pattern=re.compile(r"so's", FLAGS), replacement="so is"),
        Rule(pattern=re.compile(r"that'd've", FLAGS), replacement="that would have"),
        Rule(pattern=re.compile(r"that'd", FLAGS), replacement="that would"),
        Rule(pattern=re.compile(r"that's", FLAGS), replacement="that is"),
        Rule(pattern=re.compile(r"there'd've", FLAGS), replacement="there would have"),
        Rule(pattern=re.compile(r"there'd", FLAGS), replacement="there would"),
        Rule(pattern=re.compile(r"there's", FLAGS), replacement="there is"),
        Rule(pattern=re.compile(r"they'd've", FLAGS), replacement="they would have"),
        Rule(pattern=re.compile(r"they'd", FLAGS), replacement="they would"),
        Rule(pattern=re.compile(r"they'll've", FLAGS), replacement="they will have"),
        Rule(pattern=re.compile(r"they'll", FLAGS), replacement="they will"),
        Rule(pattern=re.compile(r"they're", FLAGS), replacement="they are"),
        Rule(pattern=re.compile(r"they've", FLAGS), replacement="they have"),
        Rule(pattern=re.compile(r"to've", FLAGS), replacement="to have"),
        Rule(pattern=re.compile(r"wasn't", FLAGS), replacement="was not"),
        Rule(pattern=re.compile(r"we'd", FLAGS), replacement="we would"),
        Rule(pattern=re.compile(r"we'd've", FLAGS), replacement="we would have"),
        Rule(pattern=re.compile(r"we'll", FLAGS), replacement="we will"),
        Rule(pattern=re.compile(r"we'll've", FLAGS), replacement="we will have"),
        Rule(pattern=re.compile(r"we're", FLAGS), replacement="we are"),
        Rule(pattern=re.compile(r"we've", FLAGS), replacement="we have"),
        Rule(pattern=re.compile(r"weren't", FLAGS), replacement="were not"),
        Rule(pattern=re.compile(r"what'll", FLAGS), replacement="what will"),
        Rule(pattern=re.compile(r"what'll've", FLAGS), replacement="what will have"),
        Rule(pattern=re.compile(r"what're", FLAGS), replacement="what are"),
        Rule(pattern=re.compile(r"what's", FLAGS), replacement="what is"),
        Rule(pattern=re.compile(r"what've", FLAGS), replacement="what have"),
        Rule(pattern=re.compile(r"when's", FLAGS), replacement="when is"),
        Rule(pattern=re.compile(r"when've", FLAGS), replacement="when have"),
        Rule(pattern=re.compile(r"where'd", FLAGS), replacement="where did"),
        Rule(pattern=re.compile(r"where's", FLAGS), replacement="where is"),
        Rule(pattern=re.compile(r"where've", FLAGS), replacement="where have"),
        Rule(pattern=re.compile(r"who'll", FLAGS), replacement="who will"),
        Rule(pattern=re.compile(r"who'll've", FLAGS), replacement="who will have"),
        Rule(pattern=re.compile(r"who's", FLAGS), replacement="who is"),
        Rule(pattern=re.compile(r"who've", FLAGS), replacement="who have"),
        Rule(pattern=re.compile(r"why's", FLAGS), replacement="why is"),
        Rule(pattern=re.compile(r"why've", FLAGS), replacement="why have"),
        Rule(pattern=re.compile(r"will've", FLAGS), replacement="will have"),
        Rule(pattern=re.compile(r"won't've", FLAGS), replacement="will not have"),
        Rule(pattern=re.compile(r"won't", FLAGS), replacement="will not"),
        Rule(pattern=re.compile(r"wouldn't've", FLAGS), replacement="would not have"),
        Rule(pattern=re.compile(r"would've", FLAGS), replacement="would have"),
        Rule(pattern=re.compile(r"wouldn't", FLAGS), replacement="would not"),
        Rule(
            pattern=re.compile(r"y'all'd've", FLAGS), replacement="you all would have"
        ),
        Rule(pattern=re.compile(r"y'all'd", FLAGS), replacement="you all would"),
        Rule(pattern=re.compile(r"y'all're", FLAGS), replacement="you all are"),
        Rule(pattern=re.compile(r"y'all've", FLAGS), replacement="you all have"),
        Rule(pattern=re.compile(r"y'all", FLAGS), replacement="you all"),
        Rule(pattern=re.compile(r"you'd've", FLAGS), replacement="you would have"),
        Rule(pattern=re.compile(r"you'd", FLAGS), replacement="you would"),
        Rule(pattern=re.compile(r"you'll've", FLAGS), replacement="you will have"),
        Rule(pattern=re.compile(r"you'll", FLAGS), replacement="you will"),
        Rule(pattern=re.compile(r"you're", FLAGS), replacement="you are"),
        Rule(pattern=re.compile(r"you've", FLAGS), replacement="you have"),
    ]
)


def expand_contractions(s: str, rules: Tuple[Rule, ...] = CONTRACTIONS) -> str:
    res = s
    for r in rules:
        res = r.pattern.sub(r.replacement, res)
    return res
