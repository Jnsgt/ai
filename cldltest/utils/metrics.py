import numpy as np


def flatten_to_numpy(x):
    """
    将输入统一转成 float32 的一维 numpy 数组
    这样不管输入原来是 list、嵌套 list，还是 numpy 数组，
    后面的误差计算都能统一处理。
    """
    return np.array(x, dtype=np.float32).reshape(-1)


def max_abs_diff(a, b):
    a = flatten_to_numpy(a)
    b = flatten_to_numpy(b)
    return float(np.max(np.abs(a - b)))


def mean_abs_diff(a, b):
    a = flatten_to_numpy(a)
    b = flatten_to_numpy(b)
    return float(np.mean(np.abs(a - b)))


def max_rel_diff(a, b, eps=1e-8):
    a = flatten_to_numpy(a)
    b = flatten_to_numpy(b)
    rel = np.abs(a - b) / (np.abs(a) + eps)
    return float(np.max(rel))


def allclose(a, b, atol=1e-6, rtol=1e-5):
    a = flatten_to_numpy(a)
    b = flatten_to_numpy(b)
    return bool(np.allclose(a, b, atol=atol, rtol=rtol))


def rmse(a, b):
    a = flatten_to_numpy(a)
    b = flatten_to_numpy(b)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def l2_distance(a, b):
    a = flatten_to_numpy(a)
    b = flatten_to_numpy(b)
    return float(np.linalg.norm(a - b))


def calc_metrics(a, b):
    """
    汇总常用误差指标，供 comparator.py 直接调用
    """
    return {
        "max_abs_diff": max_abs_diff(a, b),
        "mean_abs_diff": mean_abs_diff(a, b),
        "max_rel_diff": max_rel_diff(a, b),
        "rmse": rmse(a, b),
        "l2_distance": l2_distance(a, b),
        "allclose": allclose(a, b),
    }