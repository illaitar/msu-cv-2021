import numpy as np
from skimage.feature import ORB, match_descriptors
from skimage.color import rgb2gray
from skimage.transform import ProjectiveTransform
from skimage.transform import warp
from skimage.filters import gaussian
import skimage
from numpy.linalg import inv
import math

DEFAULT_TRANSFORM = ProjectiveTransform


def find_orb(img, n_keypoints=600):
    """Find keypoints and their descriptors in image.

    img ((W, H, 3)  np.ndarray) : 3-channel image
    n_keypoints (int) : number of keypoints to find

    Returns:
        (N, 2)  np.ndarray : keypoints
        (N, 256)  np.ndarray, type=np.bool  : descriptors
    """
    descriptor_extractor = ORB(n_keypoints = n_keypoints)
    descriptor_extractor.detect_and_extract(img[:,:,1])
    keypoints = descriptor_extractor.keypoints
    descriptors = descriptor_extractor.descriptors
    return keypoints, descriptors


def center_and_normalize_points(points):
    """Center the image points, such that the new coordinate system has its
    origin at the centroid of the image points.

    Normalize the image points, such that the mean distance from the points
    to the origin of the coordinate system is sqrt(2).

    points ((N, 2) np.ndarray) : the coordinates of the image points

    Returns:
        (3, 3) np.ndarray : the transformation matrix to obtain the new points
        (N, 2) np.ndarray : the transformed image points
    """
    matrix = np.zeros((3, 3))
    centroid = [np.mean(points[0]),np.mean(points[1])]
    points = points - centroid
    N = math.sqrt(np.sum((points[:,0] * points[:,0] + points[:,1] * points[:,1]) * 2))
    N = 1 / N
    points = points * N
    matrix[0,0] = N
    matrix[1,1] = N
    matrix[2,2] = 1
    matrix[0,2] = - N * centroid[0]
    matrix[1,2] = - N * centroid[1]
    return matrix, points


def find_homography(src_keypoints, dest_keypoints):
    """Estimate homography matrix from two sets of N (4+) corresponding points.

    src_keypoints ((N, 2) np.ndarray) : source coordinates
    dest_keypoints ((N, 2) np.ndarray) : destination coordinates

    Returns:
        ((3, 3) np.ndarray) : homography matrix
    """
    src_matrix, src = center_and_normalize_points(src_keypoints)
    dest_matrix, dest = center_and_normalize_points(dest_keypoints)
    points1 = np.column_stack((src,np.ones(src.shape[0]))).T
    points2 = np.column_stack((dest,np.ones(dest.shape[0]))).T
    p1 = points1[:-1,:].T
    p2 = points2[:-1,:].T
    A_up = np.column_stack((p1,np.ones(p1.shape[0]),np.zeros((p1.shape[0],3)),-p1[:,0]*p2[:,0],-p1[:,1]*p2[:,0],-p2[:,0]))
    A_below = np.column_stack((np.zeros((p1.shape[0],3)),p1,np.ones(p1.shape[0]),-p1[:,0]*p2[:,1],-p1[:,1]*p2[:,1],-p2[:,1]))
    A = np.vstack((A_up,A_below))
    H = np.linalg.svd(A)[-1][-1]
    H = H.reshape((p1.shape[1]+1,-1))
    H = np.linalg.inv(dest_matrix)@H@src_matrix
    return H


def ransac_transform(src_keypoints, src_descriptors, dest_keypoints, dest_descriptors, max_trials = 700, residual_threshold = 10, return_matches=False):
    """Match keypoints of 2 images and find ProjectiveTransform using RANSAC algorithm.

    src_keypoints ((N, 2) np.ndarray) : source coordinates
    src_descriptors ((N, 256) np.ndarray) : source descriptors
    dest_keypoints ((N, 2) np.ndarray) : destination coordinates
    dest_descriptors ((N, 256) np.ndarray) : destination descriptors
    max_trials (int) : maximum number of iterations for random sample selection.
    residual_threshold (float) : maximum distance for a data point to be classified as an inlier.
    return_matches (bool) : if True function returns matches

    Returns:
        skimage.transform.ProjectiveTransform : transform of source image to destination image
        (Optional)(N, 2) np.ndarray : inliers' indexes of source and destination images
    """
    np.random.seed(0)
    best_inlaers = 0
    best_points = None
    match = match_descriptors(src_descriptors,dest_descriptors)
    src_keypoints = src_keypoints[match[:,0]]
    dest_keypoints = dest_keypoints[match[:,1]]
    x = src_keypoints.shape[0]
    for zzz in range(max_trials):
        rnd = np.random.choice(np.arange(x), 4)
        src_p = src_keypoints[rnd]
        dest_p = dest_keypoints[rnd]
        H = find_homography(src_p,dest_p)
        cur = []
        err = np.sum((dest_keypoints - ProjectiveTransform(H)(src_keypoints)) ** 2, axis=1) ** 0.5
        cur =  np.arange(x) * (err < residual_threshold)
        cur = cur[cur!=0]
        if len(cur) > best_inlaers:
            best_inlaers = len(cur)
            best_points = cur.copy()
    H = find_homography(src_keypoints[best_points], dest_keypoints[best_points])
    if return_matches:
        return ProjectiveTransform(H), match[best_points]
    else:
        return ProjectiveTransform(H)


def find_simple_center_warps(forward_transforms):
    """Find transformations that transform each image to plane of the central image.

    forward_transforms (Tuple[N]) : - pairwise transformations

    Returns:
        Tuple[N + 1] : transformations to the plane of central image
    """
    len_ = len(forward_transforms) + 1
    center = (len_ - 1)//2
    res = [None] * len_
    res[center] = DEFAULT_TRANSFORM()
    for i in range(center-1,-1,-1):
        res[i] = res[i+1] + forward_transforms[i]
    for i in range(center+1,len_):
        res[i] = res[i-1] + DEFAULT_TRANSFORM(inv(forward_transforms[i-1].params))
    return tuple(res)


def get_corners(image_collection, center_warps):
    """Get corners' coordinates after transformation."""
    for img, transform in zip(image_collection, center_warps):
        height, width, _ = img.shape
        corners = np.array([[0, 0],
                            [height, 0],
                            [height, width],
                            [0, width]])
        yield transform(corners)[:, ::-1]


def get_min_max_coords(corners):
    """Get minimum and maximum coordinates of corners."""
    corners = np.concatenate(corners)
    return corners.min(axis=0), corners.max(axis=0)

def get_final_center_warps(image_collection, simple_center_warps):
    """Find final transformations.

        image_collection (Tuple[N]) : list of all images
        simple_center_warps (Tuple[N])  : transformations unadjusted for shift

        Returns:
            Tuple[N] : final transformations
        """
    corners = tuple(get_corners(image_collection, simple_center_warps))
    min_, max_ = get_min_max_coords(corners)
    array = np.array([[1, 0 ,-min_[1]],[0, 1, -min_[0]],[0, 0, 1 ]])
    Proj = ProjectiveTransform(array)
    result = list(simple_center_warps)
    for i in range(len(simple_center_warps)):
        result[i] += Proj
    return (result,(int(round(max_[1]-min_[1])),int(round(max_[0]-min_[0])),3))

def rotate_transform_matrix(transform):
    """Rotate matrix so it can be applied to row:col coordinates."""
    matrix = transform.params[(1, 0, 2), :][:, (1, 0, 2)]
    return type(transform)(matrix)


def warp_image(image, transform, output_shape):
    """Apply transformation to an image and its mask

    image ((W, H, 3)  np.ndarray) : image for transformation
    transform (skimage.transform.ProjectiveTransform): transformation to apply
    output_shape (int, int) : shape of the final pano

    Returns:
        (W, H, 3)  np.ndarray : images image
        (W, H)  np.ndarray : images mask
    """
    img = warp(image,rotate_transform_matrix(transform).inverse,mode = 'constant',output_shape=output_shape,order = 0,cval = 0 )
    mask = np.ones(image.shape[:2])
    mask = warp(mask,rotate_transform_matrix(transform).inverse,mode = 'constant',output_shape=output_shape,order=0,cval=0)
    mask = (mask>0).astype('int')
    return img, mask


def merge_pano(image_collection, final_center_warps, output_shape):
    """ Merge the whole panorama

    image_collection (Tuple[N]) : list of all images
    final_center_warps (Tuple[N])  : transformations
    output_shape (int, int) : shape of the final pano

    Returns:
        (output_shape) np.ndarray: final pano
    """
    image = np.zeros(output_shape)
    for i in range (len(image_collection)):
        img, mask = warp_image(image_collection[i],final_center_warps[i],output_shape)
        image[np.where(mask == 1)] = img[np.where(mask == 1)]
    return np.clip(image*255,0,255).astype('uint8')

def get_gaussian_pyramid(image, n_layers, sigma):
    """Get Gaussian pyramid.

    image ((W, H, 3)  np.ndarray) : original image
    n_layers (int) : number of layers in Gaussian pyramid
    sigma (int) : Gaussian sigma

    Returns:
        tuple(n_layers) Gaussian pyramid

    """
    pyr = [image]
    for i in range(n_layers-1):
        image = gaussian(image,sigma)
        pyr.append(image)
    return pyr


def get_laplacian_pyramid(image, n_layers = 4, sigma = 3):
    """Get Laplacian pyramid

    image ((W, H, 3)  np.ndarray) : original image
    n_layers (int) : number of layers in Laplacian pyramid
    sigma (int) : Gaussian sigma

    Returns:
        tuple(n_layers) Laplacian pyramid
    """
    result = []
    img = image
    gauss = get_gaussian_pyramid(image,n_layers, sigma)
    for i in range(n_layers-1):
        res = gauss[i] - gauss[i+1]
        result.append(res)
    result.append(gauss[n_layers-1])
    return result


def merge_laplacian_pyramid(laplacian_pyramid):
    """Recreate original image from Laplacian pyramid

    laplacian pyramid: tuple of np.array (h, w, 3)

    Returns:
        np.array (h, w, 3)
    """
    return sum(laplacian_pyramid)


def increase_contrast(image_collection):
    """Increase contrast of the images in collection"""
    result = []
    for img in image_collection:
        img = img.copy()
        for i in range(img.shape[-1]):
            img[:, :, i] -= img[:, :, i].min()
            img[:, :, i] /= img[:, :, i].max()
        result.append(img)
    return result


def gaussian_merge_pano(image_collection, final_center_warps, output_shape, n_layers=4, image_sigma=1, merge_sigma=3.5):
    """ Merge the whole panorama using Laplacian pyramid
    image_collection (Tuple[N]) : list of all images
    final_center_warps (Tuple[N])  : transformations
    output_shape (int, int) : shape of the final pano
    n_layers (int) : number of layers in Laplacian pyramid
    image_sigma (int) :  sigma for Gaussian filter for images
    merge_sigma (int) : sigma for Gaussian filter for masks
    Returns:
        (output_shape) np.ndarray: final pano
    """
    images = []
    masks = []
    res = np.zeros(output_shape)
    for i in range(len(image_collection)):
        img, mask = warp_image(image_collection[i], final_center_warps[i], output_shape)
        images.append(img)
        masks.append(mask)
    for i in range(len(image_collection)-1):
        cur = np.bitwise_or(masks[i], masks[i + 1])
        cur_mod = (np.bitwise_and(masks[i],masks[i + 1])).max(axis=0) > 0
        l_border = cur_mod.argmax()
        r_border = cur_mod.shape[0] - cur_mod[::-1].argmax() - 1
        center = (l_border + r_border)//2
        #print(center)
        cur = np.hstack((cur[:,:center], np.zeros((output_shape[0], output_shape[1] - center))))
        ###
        laplacian = get_laplacian_pyramid(images[i], n_layers, image_sigma)
        laplacian_next = get_laplacian_pyramid(images[i+1], n_layers, image_sigma)
        gaussian = get_gaussian_pyramid(cur, n_layers, merge_sigma)
        ###
        laplacian = np.asarray([zzz for zzz in laplacian])
        laplacian_next = np.asarray([zzz for zzz in laplacian_next])
        gaussian = np.asarray([zzz for zzz in gaussian])
        ###
        lapl = np.zeros(laplacian.shape)
        inv = 1-gaussian
        for k in range(3):
            lapl[:,:,:,k] = gaussian*laplacian[:,:,:,k] + inv * laplacian_next[:,:,:,k]
        images[i+1] = merge_laplacian_pyramid(tuple(lapl))
        masks[i+1] = np.bitwise_or(masks[i+1],masks[i])
    return np.clip(images[-1]*255,0,255).astype('uint8')

def cylindrical_inverse_map(coords, h, w, scale):
    """Function that transform coordinates in the output image
    to their corresponding coordinates in the input image
    according to cylindrical transform.

    Use it in skimage.transform.warp as `inverse_map` argument

    coords ((M, 2) np.ndarray) : coordinates of output image (M == col * row)
    h (int) : height (number of rows) of input image
    w (int) : width (number of cols) of input image
    scale (int or float) : scaling parameter

    Returns:
        (M, 2) np.ndarray : corresponding coordinates of input image (M == col * row) according to cylindrical transform
    """
    coords = np.row_stack([coords.T, np.ones((coords.shape[0]), dtype=np.float32)])
    K = np.array([[scale, 0, w//2],[0, scale, h//2],[0, 0, 1]], dtype=np.float32)
    inversed = np.linalg.inv(K)
    center = inversed@coords
    projection = center.copy()
    #pdojection.dtype
    projection[0,:] = np.tan(center[0,:])
    projection[1,:] = center[1,:]/np.cos(center[0,:])
    transformed = K@projection
    result = transformed[:2, :].T
    #print result
    #print(result.shape)
    return result

def warp_cylindrical(img, scale=None, crop=True):
    """Warp image to cylindrical coordinates

    img ((H, W, 3)  np.ndarray) : image for transformation
    scale (int or None) : scaling parameter. If None, defaults to W * 0.5
    crop (bool) : crop image to fit (remove unnecessary zero-padding of image)

    Returns:
        (H, W, 3)  np.ndarray : images image (H and W may differ from original)
    """
    h, w, _ = img.shape
    half_w = w//2
    half_h = h//2
    S = (scale, half_w)[scale is None]
    res = warp(img, cylindrical_inverse_map, map_args={'h': h,'w': w,'scale': S}, order=3)
    res = np.clip((255 * res),0,255).astype(np.uint8)
    if crop:
        gray = rgb2gray(res)
        b1 = max([j for j in range(half_h) if np.all(gray[j,:]==0)], default=0)
        b2 = min([j for j in range(half_h, h) if np.all(gray[j,:]==0)], default=half_h)
        b3 = max([i for i in range(half_w) if np.all(gray[:,i]==0)], default=0)
        b4 = min([i for i in range(half_w, w) if np.all(gray[:,i]==0)], default=half_w)
        return res[b1:b2, b3:b4, :]
    else:
        return res
