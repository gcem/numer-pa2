import dataio as io
import pa2
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    (images, labels) = io.getTrainingData()

    # io.showImage(images[0, ...], 'First image')
    covarianceMatrix = pa2.getCovarianceMatrix(images[:1000, ...])
    # svd = np.linalg.svd(covarianceMatrix)
    # singularValues = svd.S[0:50]
    largestSingularValues = np.linalg.svd(np.asmatrix(covarianceMatrix),
                                          compute_uv=False)[:50]

    ax = plt.plot(largestSingularValues, label='Singular values')
    ax = plt.plot(np.square(largestSingularValues), label='Singular values')
    ax.set_title('50 gröingular values of Y and eigenvalues of YY^T')

    plt.show(block=True)
