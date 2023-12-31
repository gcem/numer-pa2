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


def showImageIntern(image: np.ndarray, title=None):
    plt.imshow(image, cmap='gray')
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


def showImages(images: np.array):
    shape = findGridShape(len(images))
    fig, axs = plt.subplots(*shape)
    axs = axs.flatten()
    for i in range(len(images)):
        ax = axs[i]
        plt.sca(ax)
        showImageIntern(images[i])
    for i in range(len(images), np.prod(shape)):
        axs[i].set_visible(False)
    plt.show(block=False)
