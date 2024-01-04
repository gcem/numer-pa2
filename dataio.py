import numpy as np
import matplotlib.pyplot as plt
import skimage as sk


def getData(imageFile: str, labelFile: str):
    images = np.fromfile(imageFile, dtype=np.uint8)
    images = np.reshape(images[16:], [-1, 28, 28])

    labels = np.fromfile(labelFile, dtype=np.uint8)
    labels = labels[8:]

    return (images, labels)


def getTrainingData():
    return getData('train-images-idx3-ubyte', 'train-labels-idx1-ubyte')


def getTestData():
    return getData('t10k-images-idx3-ubyte', 't10k-labels-idx1-ubyte')


def getDataByLabel(data: np.ndarray, labels: np.ndarray):
    result = []
    for i in range(10):
        result.append(data[labels == i, ...])
    return result


def flatten(data: list[tuple]):
    return [elem for tup in data for elem in tup]


def showImageIntern(image: np.ndarray,
                    title: str | None = None,
                    cmap: str = 'gray',
                    vmin: float | None = None,
                    vmax: float | None = None):
    plt.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
    ax = plt.gca()
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title)


def showImage(image: np.ndarray, title=None):
    plt.figure()
    showImageIntern(image, title)
    plt.show(block=False)


def findGridShape(n: int):
    if n == 10:
        return (2, 5)
    if n == 20:
        return (4, 6)
    return (1, n)


def getImshowOptions(images: np.ndarray):
    cmap = 'gray'
    vmin = 0
    vmax = 255
    minmax = max(images.max(), abs(images.min()))
    if minmax > vmax or minmax < 2:
        vmax = minmax
    if images.min() < 0:
        cmap = 'seismic'
        vmin = -1 * vmax
    return cmap, vmin, vmax


def showImages(images: np.ndarray | list,
               indices: list | None = None,
               createFigure=True):
    images = np.array(images)
    if not indices:
        indices = range(len(images))
    cmap, vmin, vmax = getImshowOptions(images[indices])
    shape = findGridShape(len(images))
    axs = []
    if createFigure:
        _, axs = plt.subplots(*shape)
        axs = axs.flatten()
    else:
        axs = plt.gcf().axes
    for i in indices:
        ax = axs[i]
        plt.sca(ax)
        showImageIntern(images[i], cmap=cmap, vmin=vmin, vmax=vmax)
    for i in range(len(images), np.prod(shape)):
        axs[i].set_visible(False)
    plt.show(block=False)
