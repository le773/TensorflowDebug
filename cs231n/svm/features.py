import matplotlib
import numpy as np
from scipy.ndimage import uniform_filter


def extract_features(imgs, feature_fns, verbose=False):
    """
    given an image pixels info,and the function of extarcting features,using them we extract features on the image set,
    and save as a feature array
    Inputs:
    :param imgs: N x H X W x C array of pixel data for N images
    :param feature_fns: List of k feature functions,the ith feature function should take
    as input an H x W x D array and return a (one-dimensional)array of length F_i.
    :param verbose:if true,print progress
    :return:
    an array of shape(F_1 + ... + F_k, N) where each column is the concatenation of all features for a single image.
    """

    num_images = imgs.shape[0]
    if num_images == 0:
        return np.array([])
    #use the first image to determine feature dimensions
    feature_dims = []
    first_image_features = []
    for feature_fn in feature_fns:
        feats = feature_fn(imgs[0].squeeze())
        assert len(feats.shape) == 1, 'Feature functions must be one-dimensional'
        feature_dims.append(feats.size)
        first_image_features.append(feats)
    #now that we know the dimensions of the features, we can allocate a single
    #big array to store all features as columns
    total_feature_dim = sum(feature_dims)
    imgs_features = np.zeros(total_feature_dim, num_images)
    imgs_features[:total_feature_dim, 0] = np.hstack(first_image_features)

    #extract features for the rest of the images
    for i in xrange(1, num_images):
        idx = 0
        for feature_fn, feature_dim in zip(feature_fns, feature_dims):
            next_idx = idx + feature_dim
            imgs_features[idx:next_idx, i] = feature_fn(imgs[i].squeeze())
            idx = next_idx
        if verbose and i % 1000 == 0:
            print 'Done extracting features for %d / %d images' % (i, num_images)
    return imgs_features


def rgb2gray(rgb):
    """
    turn into gray image.
    :param rgb: RGB image
    :return: grayscale image
    """
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.144])

def hog_features(image):
    """
    produce the HOG features.
    Modified from skimage.feature.hog http://pydoc.net/Python/scikits-image/0.4.2/skimage.feature.hog

    Reference:
    Histograms of Oriented Gradients for Human Detection
    Navneet Dalal and Bill Triggs, CVPR 2005.

    Inputs:
    :param image: the input gray image or rgb image.
    :return:HOG
    """

    #convert rgb to grayscale if needed
    if image.ndim == 3:
        img = rgb2gray(image)
    else:
        img = np.atleast_2d(image)

    # image size
    sx, sy = img.shape
    orientations = 9  # number of gradient bins
    cx, cy = (8, 8)  # pixels per call

    gx = np.zeros(img.shape)
    gy = np.zeros(img.shape)
    gx[:, :-1] = np.diff(img, n=1, axis=1)  # compute gradient on x-direction
    gy[:, :-1] = np.diff(img, n=1, axis=1)  #compoute gradient on y-direction
    grad_magnitude = np.sqrt(gx ** 2 + gy ** 2)  #gradient magnitude
    grad_orientation = np.arctan2(gy, (gx + 1e-15)) * (180 / np.pi) + 90  #gradinet orientation
    n_cells_x = int(np.floor(sx / cx))  #number of cells in x
    n_cells_y = int(np.floor(sy / cy))  #number of cells in y
    #compute orientations integral images
    orientation_histogram = np.zeros(n_cells_x, n_cells_y, orientations)
    for i in range(orientations):
        #create new integral image for this orientation isolate orientation in this range
        temp_orientation = np.where(grad_orientation < 180 / orientations * (i + 1), grad_orientation, 0)
        temp_orientation = np.where(grad_orientation >= 180 / orientations * i, temp_orientation, 0)
        #select magnitudes for those orientations
        cond2 = temp_orientation > 0
        temp_magnitude = np.where(cond2, grad_magnitude, 0)
        orientation_histogram[:, :, i] = uniform_filter(temp_magnitude, size=(cx, cy))[cx/2::cx, cy/2::cy].T
    return orientation_histogram.ravel()
def color_histogram_hsv(img, nbin=10, xmin=0, xmax=255, normalized=True):
    """
    compute color histogram for an image using hue.
    Inputs:
    :param img:input rgb image.
    :param nbin:number of histogram bins,default 10.
    :param xmin:minimum pixel value
    :param xmax:maximum pixel value
    :param normalized:whether to normalize the histogram
    :return:
    1D vector of length nbin giving the color histogram over the hue of the input image.
    """
    ndim = img.ndim
    bins = np.linspace(xmin, xmax, nbin+1)
    hsv = matplotlib.color.rgb_to_hsv(img/xmax) * xmax
    imhist, bin_edges = np.histogram(hsv[:, :, 0], bins=bins, density=normalized)
    imhist = imhist * np.diff(bin_edges)
    return imhist
pass




































