import dataio as io
import task2
import task3
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    imageByLabel = io.getDataByLabel(*io.getTrainingData())

    two = imageByLabel[1][:1000]
    six = imageByLabel[7][:1000]

    sample = np.concatenate([two, six], axis=0)

    A, b = task3.fitPlane(sample, 2)

    subspaceCoordinates = task3.getSubspaceCoordinates(
        A, b, task2.stackImages(sample))

    ax = plt.subplot(4, 8, (2, 4 + 8 * 2))
    plt.gcf().suptitle('Aufgabe 4')

    io.scatter2(subspaceCoordinates[:1000, :], subspaceCoordinates[1000:, :],
                '1', '7')
    ax = plt.gca()
    ax.legend()
    ax.set_title('Auf $H_2$ projizierte Koordinaten für die Ziffern 2 und 6')

    ax = plt.subplot(4, 8, 3 + 8 * 3)  # middle of x axis
    io.showImage(task3.vector2image(A[:, 0]),
                 '1. Hauptkomponent',
                 createFigure=False)
    ax = plt.subplot(4, 8, 1 + 8 * 1)  # middle of y axis
    io.showImage(task3.vector2image(A[:, 1]),
                 '2. Hauptkomponent',
                 createFigure=False)

    plt.subplots_adjust(hspace=.8, wspace=.5)
    plt.show(block=True)
