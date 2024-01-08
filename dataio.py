import numpy as np
import matplotlib.pyplot as plt

#############################################################################
# Diese Datei enthält die Hilfsfunktionen für das Lesen und die Anzeige von #
# Daten.                                                                    #
#############################################################################


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
               createFigure=True,
               imshowOptions: tuple | None = None):
    images = np.array(images)
    if not indices:
        indices = range(len(images))
    cmap, vmin, vmax = imshowOptions if imshowOptions else getImshowOptions(
        images[indices])
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
    color1, color2 = 'red', 'blue'
    if data1[:, 0].mean() < data2[:, 0].mean():
        color1, color2 = color2, color1
    plt.scatter(x=data1[:, 0], y=data1[:, 1], c=color1, label=label1)
    plt.scatter(x=data2[:, 0], y=data2[:, 1], c=color2, label=label2)


def drawMiddleLine(point1: np.ndarray, point2: np.ndarray):
    left, right = plt.xlim()
    bottom, top = plt.ylim()
    midPoint = (point1 + point2) / 2
    direction = point1 - point2
    direction = np.array([-direction[1], direction[0]])  # rotate
    if direction[1] == 0:
        plt.plot([left, right], [midPoint[1], midPoint[1]])
        return
    direction /= direction[1]  # set height to 1
    t1 = (left - midPoint[0]) / direction[0]
    t2 = (right - midPoint[0]) / direction[0]
    if t2 > t1:
        t1, t2 = t2, t1
    t1 = min(t1, top - midPoint[1])
    t2 = max(t2, bottom - midPoint[1])

    start = midPoint + t1 * direction
    end = midPoint + t2 * direction
    plt.plot([start[0], end[0]], [start[1], end[1]], c="black")
    plt.xlim(left, right)
    plt.ylim(bottom, top)
