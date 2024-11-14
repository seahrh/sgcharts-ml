from typing import AnyStr

from scml.nlp import collapse_whitespace, to_str

__all__ = ["BasicPreprocessor"]


class Preprocessor:
    def __call__(self, s: AnyStr, *args, **kwargs) -> str:
        raise NotImplementedError("Implement this method in subclass")


class BasicPreprocessor(Preprocessor):
    def __init__(self):
        super().__init__()

    def __call__(self, s: AnyStr, *args, **kwargs) -> str:
        res: str = to_str(s)
        res = collapse_whitespace(res)
        return res


from .en import *

__all__ += en.__all__  # type: ignore  # module name is not defined
