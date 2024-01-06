import dataio as io
import task2
import task3
import matplotlib.pyplot as plt
import numpy as np


def findAndShowH2Projections(imageByLabel: list[np.ndarray], digit1: int,
                             digit2: int):
    data1 = imageByLabel[digit1][:1000]
    data2 = imageByLabel[digit2][:1000]
    sample = np.concatenate([data1, data2], axis=0)

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

    return subspaceCoordinates


def kMeans(coordinates: np.ndarray, mean1: np.ndarray, mean2: np.ndarray):
    # find the middle line
    class1 = np.array([])
    class2 = np.array([])

    iter = 0
    while iter < 5000:
        direction = mean2 - mean1
        middleDistance = (mean1.dot(direction) + mean2.dot(direction)) / 2
        distances = coordinates.dot(direction)
        newClass1 = distances <= middleDistance
        newClass2 = distances > middleDistance
        if np.array_equal(class1, newClass1):
            return coordinates[class1, :], coordinates[class2, :], iter
        class1 = newClass1
        class2 = newClass2
        mean1 = coordinates[class1].mean(axis=0)
        mean2 = coordinates[class2].mean(axis=0)
        iter += 1

    return coordinates[class1, :], coordinates[class2, :], iter
    # return mean1, mean2, iter


if __name__ == '__main__':
    imageByLabel = io.getDataByLabel(*io.getTrainingData())
    digit1 = 1
    digit2 = 7
    coordinates = findAndShowH2Projections(imageByLabel, digit1, digit2)
    class1, class2, iter = kMeans(coordinates, coordinates[:1000].mean(axis=0),
                                  coordinates[1000:].mean(axis=0))

    ax = plt.subplot(4, 8, (6, 3 * 8))  # middle of x axis
    io.scatter2(class1, class2, str(digit1), str(digit2))
    plt.scatter(x=[class1.mean(axis=0)[0],
                   class2.mean(axis=0)[0]],
                y=[class1.mean(axis=0)[1],
                   class2.mean(axis=0)[1]],
                c='black',
                marker='s',
                linewidths=5)
    ax.legend()
    ax.set_title(f'Klassifizierung nach {iter} K-Means-Iterationen')

    plt.show(block=True)
