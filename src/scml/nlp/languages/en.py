import warnings
from typing import AnyStr, List, Optional, Set

import scml
from scml.nlp import collapse_whitespace
from scml.nlp.charencoding import to_ascii
from scml.nlp.contractions import ContractionExpansion
from scml.nlp.findreplace import replace_punctuation
from scml.nlp.languages import Preprocessor

try:
    import spacy
except ImportError:
    spacy = None  # type: ignore
    warnings.warn("Install spacy to use this feature", ImportWarning)


__all__ = ["EnglishPreprocessor"]
log = scml.get_logger(__name__)


class EnglishPreprocessor(Preprocessor):
    def __init__(
        self,
        stopwords: Optional[Set[str]] = None,
        spacy_model_name: str = "en_core_web_sm",
    ):
        super().__init__()
        self.contraction = ContractionExpansion()
        self.nlp = spacy.load(spacy_model_name, exclude=["textcat", "ner", "tok2vec"])
        log.debug(self.nlp.pipe_names)
        self.stopwords: Set[str] = (
            self.nlp.Defaults.stop_words if stopwords is None else stopwords
        )
        log.debug(f"{len(self.stopwords)} stopwords\n{sorted(list(self.stopwords))}")

    def __call__(
        self,
        s: AnyStr,
        expand_contraction: bool = False,
        remove_punctuation: bool = False,
        lemmatize: bool = False,
        drop_stopword: bool = False,
        lowercase: bool = False,
        **kwargs,
    ) -> str:
        res: str = to_ascii(s)
        if expand_contraction:
            res = self.contraction.apply(res)
        # Expand contractions before removing punctuation
        if remove_punctuation:
            res = replace_punctuation(res, replacement=" ")
        if lemmatize:
            doc = self.nlp(res)
            tokens: List[str] = []
            for token in doc:
                t = token.lemma_
                if drop_stopword and t in self.stopwords:
                    continue
                tokens.append(t)
            res = " ".join(tokens)
        if lowercase:
            res = res.lower()
        res = collapse_whitespace(res)
        return res
