import dataio as io
import numpy as np
import matplotlib.pyplot as plt


def sampleMeanAndVariance(images: np.ndarray):
    """Findet den empirischen Mittelwert und die empirische Varianz.
    """
    mean = images.mean(axis=0)
    variance = np.square(images - mean).mean(axis=0)
    return mean, variance


def doTask1():
    imageByLabel = io.getDataByLabel(*io.getTrainingData())

    meansAndVariances = [
        sampleMeanAndVariance(imageByLabel[digit][:100, ...])
        for digit in range(10)
    ]

    # Mittelwert und Varianz anzeigen
    io.showImages(io.flatten(meansAndVariances), range(0, 20, 2))
    io.showImages(io.flatten(meansAndVariances),
                  range(1, 20, 2),
                  createFigure=False)

    # die Anzeige konfigurieren (Titel)
    ax = plt.gcf().add_subplot(4, 6, (21, 24))
    ax.set_axis_off()
    plt.text(
        0.5,
        0.5,
        'Empirischer Mittelwert (links)\nund\nempirische Varianz (rechts)',
        horizontalalignment='center',
        verticalalignment='center',
        transform=ax.transAxes,
        wrap=True,
        fontsize=16)

    plt.gcf().suptitle('Aufgabe 1')


if __name__ == '__main__':
    doTask1()
    plt.show(block=True)
