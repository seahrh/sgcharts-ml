import json

import numpy as np
import pytest

from scml.serdes import *


class TestNumpyEncoder:
    @pytest.mark.parametrize(
        "inp_list", [[[1, 2, 3], [4, 5, 6]], [[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]]]
    )
    def test_serialize_then_deserialize(self, inp_list):
        a = np.array(inp_list)
        ser: str = json.dumps(obj=a, cls=NumpyEncoder)
        assert ser == str(inp_list)
        des = json.loads(ser)
        assert type(des) is list
        np.testing.assert_allclose(a, np.array(des))

    def test_serialize_int_and_float_arrays_in_json_object(self):
        ints = np.array([[1, 2, 3], [4, 5, 6]])
        floats = np.array([[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]])
        obj = {"f1": floats, "f2": [2, (2, 3, 4), ints], "f3": [1, 2]}
        ser: str = json.dumps(obj=obj, cls=NumpyEncoder)
        assert (
            ser
            == '{"f1": [[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]], "f2": [2, [2, 3, 4], [[1, 2, 3], [4, 5, 6]]], "f3": [1, 2]}'
        )


class TestSetEncoder:
    @pytest.mark.parametrize(
        "obj,ser_1,ser_2",
        [
            pytest.param(set(), "[]", "[]", id="empty set"),
            pytest.param({"1", "a"}, '["1", "a"]', '["a", "1"]', id="non-empty set"),
            pytest.param(
                {"field": {"1", "a"}},
                '{"field": ["1", "a"]}',
                '{"field": ["a", "1"]}',
                id="dict contains set",
            ),
            pytest.param(
                {"outer_field": {"inner_field": {"1", "a"}}},
                '{"outer_field": {"inner_field": ["1", "a"]}}',
                '{"outer_field": {"inner_field": ["a", "1"]}}',
                id="nested dict contains set",
            ),
        ],
    )
    def test_serialize(self, obj, ser_1, ser_2):
        ser = json.dumps(obj=obj, cls=SetEncoder)
        assert ser == ser_1 or ser == ser_2
