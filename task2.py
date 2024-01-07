import dataio as io
import task1
import matplotlib.pyplot as plt
import numpy as np


def stackImages(images: np.ndarray):
    """Returns images as columns of a matrix
    """
    # reshape in a column-major way (-1, ...). then we have to transpose
    return images.reshape((-1, 28 * 28), order='F').transpose()


def findDeviations(images: np.ndarray):
    mean = task1.sampleMean(images)
    return images - mean


def doTask2():
    plt.figure(figsize=[12, 6])
    (images, labels) = io.getTrainingData()

    sample = images[:1000, ...]

    Y = stackImages(findDeviations(sample))

    largestSingularValues = np.linalg.svd(np.asmatrix(Y),
                                          compute_uv=False)[:50]

    S = Y @ Y.transpose()

    # since S is symmetric, use eigh
    largestEigenvalues = np.linalg.eigvalsh(S)[:-51:-1]

    plt.plot(np.square(largestSingularValues),
             label='50 größte quadrierte Singulärwerte von $Y$')
    plt.plot(largestEigenvalues, label='50 größte Eigenwerte von $YY^T$')

    ax = plt.gca()
    ax.legend()
    ax.set_title('Eigenwerte und quadrierte Singulärwerte')
    ax.grid()
    ax.set_xticks(range(50))
    ax.set_xticklabels([1] + [i if i % 5 == 0 else '' for i in range(2, 51)])
    ax.set_xlabel('Index')
    ax.set_ylabel('Eigenwert / Quadrierter Singulärwert')

    plt.gcf().suptitle('Aufgabe 2')


if __name__ == '__main__':
    doTask2()
    plt.show(block=True)
