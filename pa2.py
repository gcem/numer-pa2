import numpy as np

### utils


def image2vector(image: np.ndarray):
    # return image.flatten('C')
    return image.flatten('F')


def vector2image(vector: np.ndarray):
    # return vector.reshape(28, 28, 'C')  # TODO not hard-code the size? - slow.
    return vector.reshape(28, 28, 'F')  # TODO not hard-code the size? - slow.
