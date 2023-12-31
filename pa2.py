import numpy as np


def sampleMean(vx: np.ndarray):
    return 1 / vx.shape[0] * vx.sum(axis=0)


def sampleMeanAndVariance(vx: np.array):
    mean = sampleMean(vx)
    variance = 1 / vx.shape[0] * np.square(vx - mean).sum(axis=0)
    return (mean, variance)
