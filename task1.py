import dataio as io
import numpy as np
import matplotlib.pyplot as plt


def sampleMean(vx: np.ndarray):
    return 1 / vx.shape[0] * vx.sum(axis=0)


def sampleMeanAndVariance(vx: np.ndarray):
    mean = sampleMean(vx)
    variance = 1 / vx.shape[0] * np.square(vx - mean).sum(axis=0)
    return (mean, variance)


if __name__ == '__main__':
    imageByLabel = io.getDataByLabel(*io.getTrainingData())

    meansAndVariances = [
        sampleMeanAndVariance(imageByLabel[digit][:100, ...])
        for digit in range(10)
    ]
    io.showImages(io.flatten(meansAndVariances))

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

    plt.show(block=True)
