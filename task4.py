import dataio as io
import task2
import task3
import matplotlib.pyplot as plt
import numpy as np


def findH2Projections(imageByLabel: list[np.ndarray], digit1: int,
                      digit2: int):
    """Projiziert jeweils 1000 Bilder für die Ziffern `digit1` und `digit2` auf
    den Unterraum mit den ersten zwei Hauptkomponenten von diesen 2000 Daten.

    Gibt den 2-dimensionalen Unterraum A, b und die 2 Koordinaten der Bilder in
    diesem Unterraum zurück.

    Die Koordinaten sind in einer n x 2 Matrix zurückgegeben.
    """
    data1 = imageByLabel[digit1][:1000]
    data2 = imageByLabel[digit2][:1000]
    sample = np.concatenate([data1, data2], axis=0)

    A, b = task3.fitPlane(sample, 2)

    return A, b, task3.getSubspaceCoordinates(A, b, task2.stackImages(sample))


def showH2Projections(A: np.ndarray, subspaceCoordinates: np.ndarray,
                      digit1: int, digit2: int):
    """Zeigt die in `subspaceCoordinates` gegebenen Projektionen für die Ziffern
    `digit1` und `digit2` mit zwei unterschiedlichen Farben an.
    
    Die ersten 1000 Punkte bilden die erste Gruppe, der Rest bildet die
    zweite Gruppe. 
    """
    # scatter plot anzeigen
    ax = plt.subplot(4, 8, (2, 4 + 8 * 2))
    plt.gcf().suptitle('Aufgabe 4')

    io.scatter2(subspaceCoordinates[:1000, :], subspaceCoordinates[1000:, :],
                str(digit1), str(digit2))

    # Anzeige konfigurieren
    ax = plt.gca()
    ax.legend()
    ax.set_title(
        f'Auf $H_2$ projizierte Koordinaten\nfür die Ziffern {digit1} und {digit2}'
    )

    xlabel = 'rot: positiv\nblau: negativ'

    # Hauptkomponente in x- und y-Achsen anzeigen, damit man sieht, was die
    # projizierten Koordinaten bedeuten
    ax = plt.subplot(4, 8, 3 + 8 * 3)  # Mitte von x-Achse
    io.showImage(task3.vector2image(A[:, 0]),
                 'x-Koordinate:\n1. Hauptkomponent',
                 createFigure=False)
    ax.set_xlabel(xlabel)
    ax = plt.subplot(4, 8, 1 + 8 * 1)  # Mitte von y-Achse
    io.showImage(task3.vector2image(A[:, 1]),
                 'y-Koordinate:\n2. Hauptkomponent',
                 createFigure=False)
    ax.set_xlabel(xlabel)

    plt.subplots_adjust(hspace=.8, wspace=.5)


def kMeans(coordinates: np.ndarray, mean1: np.ndarray, mean2: np.ndarray):
    """K-Means-Algorithmus für zwei Klassen.
    
    Weil wir nur zwei Klassen haben, berechnen wir die euklid'schen Abstände
    nicht. Stattdessen sehen wir, auf welcher Seite der mittleren Hyperebene,
    die die Mittelpunkte von den 2 Klassen trennt, jeder Punkt sich befindet.

    Gibt
    * die Listen von Koordinaten in der ersten und zweiten Klasse nach dem
    Algorithmus,
    * die Mittelwerte von den beiden Klassen,
    * Anzahl der Iterationen
    zurück.
    """
    pointsInClass1 = np.array([])
    pointsInClass2 = np.array([])

    iter = 0
    while iter < 5000:
        iter += 1

        # Vektor von mean1 zu mean2
        direction = mean2 - mean1

        # Abstand der Hyperebene zum Ursprung in Richtung direction
        # (Einheit: Länge vom Vektor direction)
        middleDistance = (mean1.dot(direction) + mean2.dot(direction)) / 2

        # Abstand der Punkte zum Ursprung in Richtung direction
        distances = coordinates.dot(direction)

        newPointsInClass1 = distances <= middleDistance
        newPointsInClass2 = distances > middleDistance

        # brich ab, falls die Klassen sich nicht mehr ändern
        if np.array_equal(pointsInClass1, newPointsInClass1):
            break

        pointsInClass1 = newPointsInClass1
        pointsInClass2 = newPointsInClass2
        mean1 = coordinates[pointsInClass1].mean(axis=0)
        mean2 = coordinates[pointsInClass2].mean(axis=0)

    return coordinates[pointsInClass1, :], coordinates[
        pointsInClass2, :], mean1, mean2, iter


def showKMeansResult(class1: np.ndarray, class2: np.ndarray, mean1: np.ndarray,
                     mean2: np.ndarray, digit1: int, digit2: int, iter: int):
    """Stellt das Ergebnis vom K-Means-Algorithmus graphisch dar.

    Die (meisten) Argumente sind die Rückgaben von `kMeans()`.
    """
    # die Punkte in zwei Klassen anzeigen
    ax = plt.subplot(4, 8, (6, 3 * 8))
    io.scatter2(class1, class2, f'Klassifiziert als {digit1}',
                f'Klassifiziert als {digit2}')

    # Mittelpunkte von den Klassen anzeigen
    plt.scatter(x=[mean1[0], mean2[0]],
                y=[mean1[1], mean2[1]],
                c='black',
                marker='s',
                linewidths=5)

    # zeige die Gerade zwischen den Mittelpunkten (die "Hyperebene" in kMeans()
    # oben) an
    io.drawMiddleLine(mean1, mean2)

    ax.legend()
    ax.set_title(
        f'Klassifizierung der Trainingsbilder\nnach {iter} K-Means-Iterationen'
    )


def showTestResultsTable(A: np.ndarray,
                         b: np.ndarray,
                         mean1: np.ndarray,
                         mean2: np.ndarray,
                         testImageByLabel: list[np.ndarray],
                         digit1: int,
                         digit2: int,
                         testSize: int = 100):
    """Projiziert `testSize` Testbilder für die Ziffern `digit1`, `digit2` auf
    den Unterraum (`A`, `b`). Klassifiziert diese Projektionen mittels `mean1`
    und `mean2`. Stellt die Anzahl der richtig und falsch klassifizierten Bilder
    in einer Tabelle dar.
    """
    digit1Images = testImageByLabel[digit1][:testSize]
    digit2Images = testImageByLabel[digit2][:testSize]

    # projizieren
    digit1Coordinates = task3.getSubspaceCoordinates(
        A, b, task2.stackImages(digit1Images))
    digit2Coordinates = task3.getSubspaceCoordinates(
        A, b, task2.stackImages(digit2Images))

    # klassifizieren
    direction = mean2 - mean1
    middleDistance = (mean1.dot(direction) + mean2.dot(direction)) / 2
    digit1Correct = digit1Coordinates.dot(direction) <= middleDistance
    digit2Correct = digit2Coordinates.dot(direction) > middleDistance

    # Optionen für die Tabelle
    rowLabels = [f'Ziffer {d}' for d in (digit1, digit2)]
    colLabels = [
        'Anzahl\nTestbilder', 'Richtig\nklassifiziert', 'Falsch\nklassifiziert'
    ]
    colors = ['lightgray', 'green', 'red']

    # Tabelle anzeigen
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

    digit1 = 1
    digit2 = 7

    imageByLabel = io.getDataByLabel(*io.getTrainingData())
    A, b, coordinates = findH2Projections(imageByLabel, digit1, digit2)
    showH2Projections(A, coordinates, digit1, digit2)

    class1, class2, mean1, mean2, iter = kMeans(
        coordinates, coordinates[:1000].mean(axis=0),
        coordinates[1000:].mean(axis=0))
    showKMeansResult(class1, class2, mean1, mean2, digit1, digit2, iter)

    testImageByLabel = io.getDataByLabel(*io.getTestData())
    showTestResultsTable(A,
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
