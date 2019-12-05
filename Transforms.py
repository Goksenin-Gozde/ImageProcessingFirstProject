from matplotlib import pyplot as plt
from skimage import data, color, transform, util


def swirled_with_checkerboard():
    image = data.checkerboard()
    swirled = transform.swirl(image, rotation=0, strength=10, radius=120)

    fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2, figsize=(8, 3),
                                   sharex=True, sharey=True)

    ax0.imshow(image, cmap=plt.cm.gray)
    ax0.axis('off')
    ax1.imshow(swirled, cmap=plt.cm.gray)
    ax1.axis('off')

    plt.show()


def swirled(img):
    swirled = transform.swirl(img, rotation=0, strength=20, radius=240)

    fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2, figsize=(8, 3),
                                   sharex=True, sharey=True)

    ax0.imshow(img, cmap=plt.cm.gray)
    ax0.axis('off')
    ax1.imshow(swirled, cmap=plt.cm.gray)
    ax1.axis('off')

    plt.show()


def rescale(img):
    fig, axes = plt.subplots(nrows=1, ncols=2)

    img_rescaled = transform.rescale(img, 0.25, anti_aliasing=False)

    ax = axes.ravel()

    ax[0].imshow(img, cmap='gray')
    ax[0].set_title("Original image")

    ax[1].imshow(img_rescaled, cmap='gray')
    ax[1].set_title("Rescaled image (anti aliasing)")

    plt.tight_layout()
    plt.show()


def resize(img):
    fig, axes = plt.subplots(nrows=1, ncols=2)

    img_rescaled = transform.resize(img, (img.shape[0] // 4, img.shape[1] // 4), anti_aliasing=False)

    ax = axes.ravel()

    ax[0].imshow(img, cmap='gray')
    ax[0].set_title("Original image")

    ax[1].imshow(img_rescaled, cmap='gray')
    ax[1].set_title("Resized image (anti aliasing)")

    plt.tight_layout()
    plt.show()


def downscale(img):

    fig, axes = plt.subplots(nrows=1, ncols=2)

    img_downScaled = transform.downscale_local_mean(img, (4, 3))

    ax = axes.ravel()

    ax[0].imshow(img, cmap='gray')
    ax[0].set_title("Original image")

    ax[1].imshow(img_downScaled, cmap='gray')
    ax[1].set_title("Downscaled image (aliasing)")

    plt.tight_layout()
    plt.show()


def rotation(image):
    radius = 705
    angle = 60
    image = util.img_as_float(image)
    rotated = transform.rotate(image, angle)

    fig, axes = plt.subplots(2, figsize=(8, 8))
    ax = axes.ravel()
    ax[0].set_title("Original")
    ax[0].imshow(image)
    ax[1].set_title("Rotated")
    ax[1].imshow(rotated)

    plt.show()
