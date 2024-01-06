import numpy as np
import matplotlib.pyplot as plt


def getData(imageFile: str, labelFile: str):
    images = np.fromfile(imageFile, dtype=np.uint8)
    images = np.reshape(images[16:], [-1, 28, 28])

    labels = np.fromfile(labelFile, dtype=np.uint8)
    labels = labels[8:]

    return (images.astype(float), labels)


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


def showImage(image: np.ndarray, title=None, createFigure=True):
    if createFigure:
        plt.figure()
    cmap, vmin, vmax = getImshowOptions(image)
    showImageIntern(image, title, cmap=cmap, vmin=vmin, vmax=vmax)
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


def scatter2(data1: np.ndarray, data2: np.ndarray, label1: str | None,
             label2: str | None):
    plt.scatter(x=data1[:, 0], y=data1[:, 1], c='red', label=label1)
    plt.scatter(x=data2[:, 0], y=data2[:, 1], c='blue', label=label2)
