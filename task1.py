import dataio as io
import pa2
import matplotlib.pyplot as plt

if __name__ == '__main__':
    imageByLabel = io.getDataByLabel(*io.getTrainingData())

    meansAndVariances = [
        pa2.sampleMeanAndVariance(imageByLabel[digit][:100, ...])
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
        fontsize=18)

    plt.show(block=True)
