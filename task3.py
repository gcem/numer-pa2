import dataio as io
import task2
import matplotlib.pyplot as plt
import numpy as np


def image2vector(image: np.ndarray):
    """Konvertiert: Bild -> Spaltenvektor
    """
    return image.flatten('F')


def vector2image(vector: np.ndarray):
    """Konvertiert: Spaltenvektor -> Bild
    """
    return vector.reshape((28, 28), order='F')


def fitPlane(sample: np.ndarray, dimension: int):
    """Findet eine `dimension`-dimensionale (affine) Ebene so, dass die Summe
    der quadrierten Abstände von Punkten in `sample` zu der Ebene minimiert ist.

    Gibt A, b zurück.
    """
    mean = sample.mean(axis=0)

    Y = task2.stackImages(task2.findDeviations(sample))
    U, sig, Vh = np.linalg.svd(Y)

    # Spalten von U sind die Eigenvektoren von S=YY^T
    return U[:, :dimension], image2vector(mean)


def getSubspaceCoordinates(A: np.ndarray, b: np.ndarray, vectors: np.ndarray):
    """Projiziert jede Spalte von `vectors` auf den Unterraum (`A`, `b`). Gibt
    die Koordinaten bzgl. der Spalten von A um den Ursprung `b` zurück.

    Gibt eine Zeile für jede Spalte von `vectors` zurück. (d.h. die i-te Spalte
    der Rückgabe enthält die i-ten Koordinaten: die Koordinaten in die Richtung
    der i-ten Spalte von A.)
    """
    # workaround: b.transpose() ist ein noop für 1D arrays
    return (vectors.transpose() - b.transpose()) @ A


def projectToAffineSubspace(A: np.ndarray, b: np.ndarray, vectors: np.ndarray):
    """Gibt die Projektion jeder Spalte von `vectors` auf den Unterraum
    (`A`, `b`) zurück.
    
    Die Projektionen sind wieder Spalten der zurückgegebenen Matrix. 
    """
    coordinates = getSubspaceCoordinates(A, b, vectors)
    return b + A @ coordinates.transpose()


def doTask3():
    (images, labels) = io.getTrainingData()

    sample = images[:1000, ...]

    A, b = fitPlane(sample, 5)

    # Mittelwert anzeigen
    io.showImage(vector2image(b), 'Mittelwert der ersten 1000 Bilder')
    plt.gcf().suptitle('Aufgabe 3')

    # Hauptkomponente anzeigen
    io.showImages([vector2image(column) for column in A.transpose()])
    plt.gcf().suptitle('Aufgabe 3')

    plt.gcf().axes[2].set_title(
        'Die ersten 5 Hauptkomponente\nfür die 1000 Trainingsbilder')
    plt.gcf().axes[2].set_xlabel('rot: positiv\nblau: negativ')

    # 5 Bilder projizieren
    testImages, testLabels = io.getTestData()
    projections = np.zeros_like(testImages[:5])
    for i in range(5):
        image = testImages[i]
        projection = projectToAffineSubspace(A, b, image2vector(image))
        projections[i] = vector2image(projection)

    # Projektionen anzeigen
    io.showImages(np.concatenate([testImages[:5], projections], axis=0),
                  imshowOptions=('RdGy', -255, 255))
    plt.gcf().suptitle('Aufgabe 3')

    plt.gcf().axes[2].set_title('5 Testbilder')
    plt.gcf().axes[2].set_xlabel('schwarz: positiv')
    plt.gcf().axes[2 + 5].set_title('Projektionen auf $H_5$')
    plt.gcf().axes[2 + 5].set_xlabel('schwarz: positiv\nrot: negativ')


if __name__ == '__main__':
    doTask3()
    plt.show(block=True)
