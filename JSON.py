"""This is the 'JSON En-/De- Coder'."""


from NNData import NNData
from collections import deque
import numpy as np
import json


class NNDataEncoder(json.JSONEncoder):

    def default(self, o):
        """Override JSON default encoder."""
        if isinstance(o, deque):
            return {"__deque__": list(o)}
        elif isinstance(o, np.ndarray):
            return {"__ndarray__": o.tolist()}
        elif isinstance(o, NNData):
            return {"NNData": o.__dict__}
        else:
            return super().default(o)


def nndata_decoder(o):
    """Implement NNData decoder."""
    if "__deque__" in o:
        return deque(o["__deque__"])
    if "__ndarray__" in o:
        return np.array(o["__ndarray__"])
    if "__NNData__" in o:
        nndata_obj = o["__NNData__"]
        return_obj = NNData()
        return_obj.features = np.array(nndata_obj["_features"])
        return_obj.labels = np.array(nndata_obj["_labels"])
        return_obj.train_factor = nndata_obj["_train_factor"]
        return_obj.train_indices = list(nndata_obj["_train_indices"])
        return_obj.test_indices = list(nndata_obj["_test_indices"])
        return_obj.train_pool = deque(nndata_obj["_train_pool"])
        return_obj.test_pool = deque(nndata_obj["_test_pool"])
        return return_obj
    else:
        return o
