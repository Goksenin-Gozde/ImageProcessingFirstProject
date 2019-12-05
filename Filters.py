from skimage import data, io, filters, feature, color, segmentation, util
from skimage.future import graph
from matplotlib import pyplot as plt
import cv2
import numpy as np
from skimage.morphology import disk, watershed
from skimage.transform import match_histograms
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed


#Filter 1
def sobel_h_filter(img):
    corners = filters.sobel_h(img)

    fig, ax = plt.subplots(nrows=2)

    ax[0].imshow(img, cmap='gray')
    ax[0].set_title("Original image")

    ax[1].imshow(corners)
    ax[1].set_title("Image with Sobel_h filter")

    plt.tight_layout()
    plt.show()


# Filter 2
def region_boundry(img):
    gimg = color.rgb2gray(img)

    fig, ax = plt.subplots(nrows=1)

    labels = segmentation.slic(img, compactness=30, n_segments=400)
    edges = filters.sobel(gimg)
    edges_rgb = color.gray2rgb(edges)

    g = graph.rag_boundary(labels, edges)
    lc = graph.show_rag(labels, g, edges_rgb, img_cmap=None, edge_cmap='viridis',
                        edge_width=1.2)

    ax.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
    ax.set_title("Original Image")

    plt.colorbar(lc, fraction=0.03)

    plt.tight_layout()
    plt.show()

# Filter 3
def try_all_threshold_filter(img):
    new_img = filters.try_all_threshold(img, figsize=(10, 6), verbose=True)
    plt.show()


# Filter 4
def roberts_filter(img):
    corners = filters.roberts(img)

    fig, ax = plt.subplots(nrows=2)

    ax[0].imshow(img, cmap='gray')
    ax[0].set_title("Original image")

    ax[1].imshow(corners)
    ax[1].set_title("Image with Roberts filter")

    plt.tight_layout()
    plt.show()


def identity(image, **kwargs):
    """Return the original image, ignoring any kwargs."""
    return image


# Filter 5
def ridge_operations(img, x1=200, x2=600, y1=400, y2=800):
    image = color.rgb2gray(img)[x1:x2, y1:y2]
    cmap = plt.cm.gray

    kwargs = {}
    kwargs['sigmas'] = [1]

    fig, axes = plt.subplots(2, 5)
    for i, black_ridges in enumerate([1, 0]):
        for j, func in enumerate([identity, filters.meijering, filters.sato, filters.frangi, filters.hessian]):
            kwargs['black_ridges'] = black_ridges
            result = func(image, **kwargs)
            if func in (filters.meijering, filters.frangi):
                result = result[4:-4, 4:-4]
            axes[i, j].imshow(result, cmap=cmap, aspect='auto')
            if i == 0:
                axes[i, j].set_title(['Original\nimage', 'Meijering\nneuriteness',
                                      'Sato\ntubeness', 'Frangi\nvesselness',
                                      'Hessian\nvesselness'][j])
            if j == 0:
                axes[i, j].set_ylabel('black_ridges = ' + str(bool(black_ridges)))
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])

    plt.tight_layout()
    plt.show()


# Filter 6
def hysteresis_threshold(img):
    fig, ax = plt.subplots(nrows=3)
    sobel_img = filters.sobel(img)
    new_img = filters.apply_hysteresis_threshold(sobel_img, 0.1, 0.35)

    ax[0].imshow(img, cmap='gray')
    ax[0].set_title('Original Image')

    ax[1].imshow(sobel_img)
    ax[1].set_title('Sobel image')

    ax[2].imshow(new_img, cmap='magma')
    ax[2].set_title('Hyteresis Threshold')

    plt.tight_layout()
    plt.show()


# Filter 7

def median_filter(img, diskValue=5):
    med = filters.median(img, disk(5))

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    med = cv2.cvtColor(med, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots(nrows=2)

    ax[0].imshow(img)
    ax[0].set_title("Original image")

    ax[1].imshow(med)
    ax[1].set_title("Image with Median filter")

    plt.tight_layout()
    plt.show()


# Filter 8
def segmentation_and_superpixel_algorithms():
    img = util.img_as_float(data.astronaut()[::2, ::2])

    segments_fz = felzenszwalb(img, scale=100, sigma=0.5, min_size=50)
    segments_slic = slic(img, n_segments=250, compactness=10, sigma=1)
    segments_quick = quickshift(img, kernel_size=3, max_dist=6, ratio=0.5)
    gradient = filters.sobel(color.rgb2gray(img))
    segments_watershed = watershed(gradient, markers=250, compactness=0.001)


    fig, ax = plt.subplots(2, 2, figsize=(10, 10), sharex=True, sharey=True)

    ax[0, 0].imshow(segmentation.mark_boundaries(img, segments_fz))
    ax[0, 0].set_title("Felzenszwalbs's method")
    ax[0, 1].imshow(segmentation.mark_boundaries(img, segments_slic))
    ax[0, 1].set_title('SLIC')
    ax[1, 0].imshow(segmentation.mark_boundaries(img, segments_quick))
    ax[1, 0].set_title('Quickshift')
    ax[1, 1].imshow(segmentation.mark_boundaries(img, segments_watershed))
    ax[1, 1].set_title('Compact watershed')

    for a in ax.ravel():
        a.set_axis_off()

    plt.tight_layout()
    plt.show()


# Filter 9

def scharr_filter(img):
    new_img = filters.scharr(img)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    new_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots(nrows=2)

    ax[0].imshow(cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB))
    ax[0].set_title("Original image")

    ax[1].imshow(cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB))
    ax[1].set_title("Image with Scharr filter")

    plt.tight_layout()
    plt.show()


# Filter 10

def find_regular_segments(img):
    original_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    edges = filters.sobel(img)

    grid = util.regular_grid(img.shape, n_points=468)

    seeds = np.zeros(img.shape, dtype=int)
    seeds[grid] = np.arange(seeds[grid].size).reshape(seeds[grid].shape) + 1

    w0 = watershed(edges, seeds)
    w1 = watershed(edges, seeds, compactness=0.01)

    fig, (ax0, ax1, ax2) = plt.subplots(1, 3)

    ax0.imshow(original_img)
    ax0.set_title("Original image")

    ax1.imshow(color.label2rgb(w0, img))
    ax1.set_title('Classical watershed')

    ax2.imshow(color.label2rgb(w1, img))
    ax2.set_title('Compact watershed')

    plt.show()


def histogram_matching():
    img = data.chelsea()
    reference = data.coffee()

    matched = match_histograms(img, reference, multichannel=True)

    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(8, 3),
                                        sharex=True, sharey=True)
    for aa in (ax1, ax2, ax3):
        aa.set_axis_off()

    ax1.imshow(img)
    ax1.set_title('Source')
    ax2.imshow(reference)
    ax2.set_title('Reference')
    ax3.imshow(matched)
    ax3.set_title('Matched')

    plt.tight_layout()
    plt.show()
