import matplotlib.pyplot as plt
from skimage import data, io, morphology
import skimage.util as util
from skimage.filters import rank
from scipy import ndimage as ndi
import numpy as np


def plot_comparison(original, filtered, filter_name):
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 4), sharex=True,
                                   sharey=True)
    ax1.imshow(original, cmap=plt.cm.gray)
    ax1.set_title('original')
    ax1.axis('off')
    ax2.imshow(filtered, cmap=plt.cm.gray)
    ax2.set_title(filter_name)
    ax2.axis('off')


# Morp 1
def erosion(img):
    orig_img = util.img_as_ubyte(img)
    selem = morphology.disk(6)
    eroded = morphology.erosion(orig_img, selem)
    plot_comparison(orig_img, eroded, 'erosion')
    plt.tight_layout()
    plt.show()


# Morp 2
def dilation(img):
    orig_img = util.img_as_ubyte(img)
    selem = morphology.disk(6)
    eroded = morphology.dilation(orig_img, selem)
    plot_comparison(orig_img, eroded, 'Dilation')
    plt.tight_layout()
    plt.show()


# Morp 3
def opening(img):
    orig_img = util.img_as_ubyte(img)
    selem = morphology.disk(6)
    eroded = morphology.opening(orig_img, selem)
    plot_comparison(orig_img, eroded, 'Opening')
    plt.tight_layout()
    plt.show()


# Morp 4
def closing(img):
    orig_img = util.img_as_ubyte(img)
    selem = morphology.disk(6)
    eroded = morphology.closing(orig_img, selem)
    plot_comparison(orig_img, eroded, 'Closing')
    plt.tight_layout()
    plt.show()


# Morp 5
def white_tophat(img):
    orig_img = util.img_as_ubyte(img)
    image = orig_img.copy()
    image[340:350, 200:210] = 255
    image[100:110, 200:210] = 0
    selem = morphology.disk(6)
    eroded = morphology.white_tophat(image, selem)
    plot_comparison(orig_img, eroded, 'White Tophat')
    plt.tight_layout()
    plt.show()


# Morp 6
def black_tophat(img):
    orig_img = util.img_as_ubyte(img)
    image = orig_img.copy()
    image[340:350, 200:210] = 255
    image[100:110, 200:210] = 0
    selem = morphology.disk(6)
    eroded = morphology.black_tophat(image, selem)
    plot_comparison(orig_img, eroded, 'Black Tophat')
    plt.tight_layout()
    plt.show()


# Morp 7
def skeletonize(img):
    eroded = morphology.skeletonize(img == 0)
    plot_comparison(img, eroded, 'Skeletonize')
    plt.tight_layout()
    plt.show()


# Morp 8
def convex_hull_image(img):
    eroded = morphology.convex_hull_image(img == 0)
    plot_comparison(img, eroded, 'Convex Hull')
    plt.tight_layout()
    plt.show()


# Morph 9

def watershed(img):
    denoised = rank.median(img, morphology.disk(2))

    markers = rank.gradient(denoised, morphology.disk(5)) < 10
    markers = ndi.label(markers)[0]

    gradient = rank.gradient(denoised, morphology.disk(2))

    labels = morphology.watershed(gradient, markers)
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 8),
                             sharex=True, sharey=True)
    ax = axes.ravel()

    ax[0].imshow(img, cmap=plt.cm.gray)
    ax[0].set_title("Original")

    ax[1].imshow(gradient, cmap=plt.cm.nipy_spectral)
    ax[1].set_title("Local Gradient")

    ax[2].imshow(markers, cmap=plt.cm.nipy_spectral)
    ax[2].set_title("Markers")

    ax[3].imshow(img, cmap=plt.cm.gray)
    ax[3].imshow(labels, cmap=plt.cm.nipy_spectral, alpha=.7)
    ax[3].set_title("Segmented")

    for a in ax:
        a.axis('off')

    fig.tight_layout()
    plt.show()


# morph 10
def entropy(img):
    noise_mask = np.full((128, 128), 28, dtype=np.uint8)
    noise_mask[32:-32, 32:-32] = 30

    noise = (noise_mask * np.random.random(noise_mask.shape) - 0.5 *
             noise_mask).astype(np.uint8)
    img = noise + 128

    entr_img = rank.entropy(img, morphology.disk(10))

    fig, (ax0, ax1, ax2) = plt.subplots(nrows=1, ncols=3, figsize=(10, 4))

    ax0.imshow(noise_mask, cmap='gray')
    ax0.set_xlabel("Noise mask")
    ax1.imshow(img, cmap='gray')
    ax1.set_xlabel("Noisy image")
    ax2.imshow(entr_img, cmap='viridis')
    ax2.set_xlabel("Local entropy")

    fig.tight_layout()

    fig.tight_layout()

    plt.show()
