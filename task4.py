import dataio as io
import task2
import task3
import matplotlib.pyplot as plt
import numpy as np


def findH2Projections(imageByLabel: list[np.ndarray], digit1: int,
                      digit2: int):
    data1 = imageByLabel[digit1][:1000]
    data2 = imageByLabel[digit2][:1000]
    sample = np.concatenate([data1, data2], axis=0)

    A, b = task3.fitPlane(sample, 2)

    return A, b, task3.getSubspaceCoordinates(A, b, task2.stackImages(sample))


def showH2Projections(A: np.ndarray, subspaceCoordinates: np.ndarray,
                      digit1: int, digit2: int):
    ax = plt.subplot(4, 8, (2, 4 + 8 * 2))
    plt.gcf().suptitle('Aufgabe 4')

    io.scatter2(subspaceCoordinates[:1000, :], subspaceCoordinates[1000:, :],
                '1', '7')
    ax = plt.gca()
    ax.legend()
    ax.set_title(
        f'Auf $H_2$ projizierte Koordinaten\nfür die Ziffern {digit1} und {digit2}'
    )

    xlabel = 'rot: positiv\nblau: negativ'
    ax = plt.subplot(4, 8, 3 + 8 * 3)  # middle of x axis
    io.showImage(task3.vector2image(A[:, 0]),
                 'x-Koordinate:\n1. Hauptkomponent',
                 createFigure=False)
    ax.set_xlabel(xlabel)
    ax = plt.subplot(4, 8, 1 + 8 * 1)  # middle of y axis
    io.showImage(task3.vector2image(A[:, 1]),
                 'y-Koordinate:\n2. Hauptkomponent',
                 createFigure=False)
    ax.set_xlabel(xlabel)

    plt.subplots_adjust(hspace=.8, wspace=.5)


def kMeans(coordinates: np.ndarray, mean1: np.ndarray, mean2: np.ndarray):
    # find the middle line
    pointsInClass1 = np.array([])
    pointsInClass2 = np.array([])

    iter = 0
    while iter < 5000:
        iter += 1
        direction = mean2 - mean1
        middleDistance = (mean1.dot(direction) + mean2.dot(direction)) / 2
        distances = coordinates.dot(direction)
        newPointsInClass1 = distances <= middleDistance
        newPointsInClass2 = distances > middleDistance
        if np.array_equal(pointsInClass1, newPointsInClass1):
            break
        pointsInClass1 = newPointsInClass1
        pointsInClass2 = newPointsInClass2
        mean1 = coordinates[pointsInClass1].mean(axis=0)
        mean2 = coordinates[pointsInClass2].mean(axis=0)

    return coordinates[pointsInClass1, :], coordinates[
        pointsInClass2, :], mean1, mean2, iter


def showKMeansClassification(class1: np.ndarray, class2: np.ndarray,
                             mean1: np.ndarray, mean2: np.ndarray, digit1: int,
                             digit2: int, iter: int):
    ax = plt.subplot(4, 8, (6, 3 * 8))  # middle of x axis
    io.scatter2(class1, class2, f'Klassifiziert als {digit1}',
                f'Klassifiziert als {digit2}')
    plt.scatter(x=[mean1[0], mean2[0]],
                y=[mean1[1], mean2[1]],
                c='black',
                marker='s',
                linewidths=5)
    io.drawMiddleLine(mean1, mean2)
    plt.plot([])
    ax.legend()
    ax.set_title(
        f'Klassifizierung der Trainingsbilder\nnach {iter} K-Means-Iterationen'
    )


def showClassificationResults(A: np.ndarray,
                              b: np.ndarray,
                              mean1: np.ndarray,
                              mean2: np.ndarray,
                              testImageByLabel: list[np.ndarray],
                              digit1: int,
                              digit2: int,
                              testSize: int = 100):
    digit1Images = testImageByLabel[digit1][:testSize]
    digit2Images = testImageByLabel[digit2][:testSize]

    digit1Coordinates = task3.getSubspaceCoordinates(
        A, b, task2.stackImages(digit1Images))
    digit2Coordinates = task3.getSubspaceCoordinates(
        A, b, task2.stackImages(digit2Images))

    direction = mean2 - mean1
    middleDistance = (mean1.dot(direction) + mean2.dot(direction)) / 2
    digit1Correct = digit1Coordinates.dot(direction) <= middleDistance
    digit2Correct = digit2Coordinates.dot(direction) > middleDistance

    rowLabels = [f'Ziffer {d}' for d in (digit1, digit2)]
    colLabels = [
        'Anzahl\nTestbilder', 'Richtig\nklassifiziert', 'Falsch\nklassifiziert'
    ]
    colors = ['lightgray', 'green', 'red']

    ax = plt.subplot(4, 8, (6 + 3 * 8, 4 * 8))
    ax.axis('off')

    table = plt.table(
        [[testSize,
          digit1Correct.sum(), testSize - digit1Correct.sum()],
         [testSize,
          digit2Correct.sum(), testSize - digit2Correct.sum()]],
        rowLabels=rowLabels,
        colLabels=colLabels,
        colColours=colors,
        bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    ax.set_title('Klassifizierung der Testbilder')


def doTask4():
    plt.figure(figsize=[12, 9])

    imageByLabel = io.getDataByLabel(*io.getTrainingData())
    digit1 = 1
    digit2 = 7
    A, b, coordinates = findH2Projections(imageByLabel, digit1, digit2)
    showH2Projections(A, coordinates, digit1, digit2)

    class1, class2, mean1, mean2, iter = kMeans(
        coordinates, coordinates[:1000].mean(axis=0),
        coordinates[1000:].mean(axis=0))
    showKMeansClassification(class1, class2, mean1, mean2, digit1, digit2,
                             iter)

    testImageByLabel = io.getDataByLabel(*io.getTestData())
    showClassificationResults(A,
                              b,
                              mean1,
                              mean2,
                              testImageByLabel,
                              digit1,
                              digit2,
                              testSize=100)


if __name__ == '__main__':
    doTask4()
    plt.show(block=True)
