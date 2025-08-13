from json import JSONEncoder

import numpy as np

__all__ = ["NumpyEncoder", "SetEncoder"]


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


class SetEncoder(JSONEncoder):
    """Encodes python set as json array.

    Based on https://stackoverflow.com/questions/8230315/how-to-json-serialize-sets
    """

    def default(self, obj):
        if isinstance(obj, set):
            return list(obj)
        return JSONEncoder.default(self, obj)
