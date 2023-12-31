import dataio as io
import pa2
import matplotlib.pyplot as plt

if __name__ == '__main__':
    (images, labels) = io.getTrainingData()

    io.showImage(images[0, ...], 'First image')

    imageByLabel = io.getDataByLabel(images, labels)

    meansAndVariances = [
        pa2.sampleMeanAndVariance(imageByLabel[digit][:100, ...])
        for digit in range(10)
    ]
    io.showImages(io.flatten(meansAndVariances))

    plt.show(block=True)
