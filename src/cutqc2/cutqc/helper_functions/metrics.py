import numpy as np
import copy
from sklearn.linear_model import LinearRegression
from qiskit.quantum_info import Statevector


def MSE(target, obs):
    """
    Mean Square Error
    """
    if isinstance(target, dict):
        se = 0
        for t_idx in target:
            t = target[t_idx]
            o = obs[t_idx]
            se += (t - o) ** 2
        mse = se / len(obs)
    elif isinstance(target, np.ndarray):
        squared_diff = (target - obs) ** 2
        mse = np.mean(squared_diff)
    elif isinstance(target, np.ndarray) and isinstance(obs, dict):
        se = 0
        for o_idx in obs:
            o = obs[o_idx]
            t = target[o_idx]
            se += (t - o) ** 2
        mse = se / len(obs)
    else:
        raise Exception("target type : %s" % type(target))
    return mse
