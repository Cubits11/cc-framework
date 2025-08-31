import numpy as np


def frechet_upper(rocA: np.ndarray, rocB: np.ndarray, comp: str = "AND") -> float:
    fA, tA = rocA[:, 0], rocA[:, 1]
    fB, tB = rocB[:, 0], rocB[:, 1]
    FA, FB = fA[:, None], fB[None, :]
    TA, TB = tA[:, None], tB[None, :]

    if comp.upper() == "AND":
        upper = np.minimum(TA, TB) - np.maximum(FA + FB - 1.0, 0.0)
    else:
        upper = np.minimum(TA + TB, 1.0) - np.maximum(FA, FB)
    return float(upper.max())
