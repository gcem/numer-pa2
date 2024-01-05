import dataio as io
import task2
import task3
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    imageByLabel = io.getDataByLabel(*io.getTrainingData())

    two = imageByLabel[2][:1000]
    six = imageByLabel[6][:1000]

    sample = np.concatenate([two, six], axis=0)

    A, b = task3.fitPlane(sample, 2)

    subspaceCoordinates = task3.getSubspaceCoordinates(
        A, b, task2.stackImages(sample))

    plt.scatter(x=subspaceCoordinates[:1000, 0],
                y=subspaceCoordinates[:1000, 1],
                c='orange',
                label='2')
    plt.scatter(x=subspaceCoordinates[1000:, 0],
                y=subspaceCoordinates[1000:, 1],
                c='cyan',
                label='6')
    ax = plt.gca()
    ax.legend()
    ax.set_title('Auf $H_2$ projizierte Koordinaten für die Ziffern 2 und 6')

    plt.gcf().suptitle('Aufgabe 4')

    plt.show(block=True)
