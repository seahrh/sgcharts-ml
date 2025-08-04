from json import JSONEncoder

import numpy as np

__all__ = ["NumpyEncoder", "NamedTupleEncoder"]


class NumpyEncoder(JSONEncoder):
    """
    Serialize numpy arrays to python lists in JSON.

    ser = json.dumps(json_obj, cls=NumpyEncoder)

    des = np.asarray(json.loads(ser))

    Based on https://stackoverflow.com/a/47626762/519951
    """

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


class NamedTupleEncoder(JSONEncoder):
    """
    Based on https://stackoverflow.com/questions/5906831/serializing-a-python-namedtuple-to-json
    """

    def _iterencode(self, obj, markers=None):
        if isinstance(obj, tuple) and hasattr(obj, "_asdict"):
            gen = self._iterencode_dict(obj._asdict(), markers)
        else:
            gen = JSONEncoder._iterencode(self, obj, markers)
        for chunk in gen:
            yield chunk
