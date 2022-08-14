import numpy as np
import orjson
import pandas as pd
import decimal


def encode_complex(obj):
    if isinstance(obj, decimal.Decimal):
        return float(obj)
    if np.isnan(obj):
        return None
    if obj in [pd.NaT, np.nan]:
        return None
    if isinstance(obj, np.int64):
        return int(obj)
    if isinstance(obj, np.float64):
        return float(obj)
    if isinstance(obj, complex):
        return obj.real
    if isinstance(obj, np.ma.core.MaskedConstant):
        return None
    raise TypeError(repr(obj) + " (Type: %s) is not JSON serializable" % type(obj))


class JsonLoader:
    @staticmethod
    def loads(string):
        return orjson.loads(string)

    @staticmethod
    def dumps(json_str):
        return orjson.dumps(json_str, default=encode_complex)


json = JsonLoader()
