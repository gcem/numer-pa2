import numpy as np

### task 1


def sampleMean(vx: np.ndarray):
    return 1 / vx.shape[0] * vx.sum(axis=0)


def sampleMeanAndVariance(vx: np.ndarray):
    mean = sampleMean(vx)
    variance = 1 / vx.shape[0] * np.square(vx - mean).sum(axis=0)
    return (mean, variance)


### task 2


def image2vector(image: np.ndarray):
    # return image.flatten('C')
    return image.flatten('F')


def vector2image(vector: np.ndarray):
    # return vector.reshape(28, 28, 'C')  # TODO not hard-code the size? - slow.
    return vector.reshape(28, 28, 'F')  # TODO not hard-code the size? - slow.


def getCovarianceMatrix(vx: np.ndarray):
    mean = sampleMean(vx)
    diffs = vx - mean
    return diffs.reshape((28 * 28, -1, 1), order='F')
