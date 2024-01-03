import dataio as io
import task1
import task2
import matplotlib.pyplot as plt
import numpy as np


def fitPlane(sample: np.ndarray, dimension: int):
    mean = task1.sampleMean(sample)

    Y = task2.stackImages(task2.findDeviations(sample))
    U, sig, Vh = np.linalg.svd(Y)

    # columns of U are the eigenvectors of S=YY^T
    return U[:, :dimension], mean


def image2vector(image: np.ndarray):
    # return image.flatten('C')
    return image.flatten('F')


def vector2image(vector: np.ndarray):
    # return vector.reshape((28, 28), order='C')
    return vector.reshape((28, 28), order='F')


if __name__ == '__main__':
    (images, labels) = io.getTrainingData()

    sample = images[:1000, ...]

    A, b = fitPlane(sample, 5)

    io.showImage(b, 'Mittelwert der ersten 1000 Bilder')

    io.showImages([vector2image(column) for column in A.transpose()])

    # TODO: these have negative values, use another color map
    plt.gcf().axes[2].set_title('Die ersten 5 Hauptkomponente')

    plt.show(block=True)
