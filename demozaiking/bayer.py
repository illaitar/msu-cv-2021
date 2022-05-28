import numpy as np
from scipy.ndimage import convolve
from math import log10, sqrt

def get_bayer_masks(n_rows, n_cols):
    a = np.zeros((n_rows // 2 * 2,n_cols // 2 * 2,3),dtype = 'bool')
    a[:,:,0] = np.tile(np.array([[False,True],[False,False]]), [n_rows//2, n_cols//2])
    a[:,:,1] = np.tile(np.array([[True,False],[False,True]]), [n_rows//2, n_cols//2])
    a[:,:,2] = np.tile(np.array([[False,False],[True,False]]), [n_rows//2, n_cols//2])
    if n_rows % 2 == 1:
        dop = a[:,0,:]
        dop.shape = dop.shape[0], 1, 3
        a = np.hstack([a,dop])
    if n_cols % 2 == 1:
        dop = a[0,:,:]
        dop.shape = 1, dop.shape[0], 3
        a = np.vstack([a,dop])
    return a
    
def get_bayer_pattern_masks(n_rows, n_cols):
    a = np.zeros((n_rows // 2 * 2,n_cols // 2 * 2,4),dtype = 'bool')
    a[:,:,0] = np.tile(np.array([[False,True],[False,False]]), [n_rows//2, n_cols//2])
    a[:,:,1] = np.tile(np.array([[False,False],[False,True]]), [n_rows//2, n_cols//2])
    a[:,:,2] = np.tile(np.array([[True,False],[False,False]]), [n_rows//2, n_cols//2])
    a[:,:,3] = np.tile(np.array([[False,False],[True,False]]), [n_rows//2, n_cols//2])
    if n_rows % 2 == 1:
        dop = a[:,0,:]
        dop.shape = dop.shape[0], 1, 4
        a = np.hstack([a,dop])
    if n_cols % 2 == 1:
        dop = a[0,:,:]
        dop.shape = 1, dop.shape[0], 4
        a = np.vstack([a,dop])
    return a[:,:,0],a[:,:,1],a[:,:,2],a[:,:,3]

def get_colored_img(raw_img):
    size_x = raw_img.shape[0]
    size_y = raw_img.shape[1]
    output = np.zeros((size_x,size_y,3),dtype = 'uint8')
    tt = get_bayer_masks(size_x, size_y)
    output[:,:,0] = tt[:,:,0] * raw_img
    output[:,:,1] = tt[:,:,1] * raw_img
    output[:,:,2] = tt[:,:,2] * raw_img
    return output
    
def bilinear_interpolation(colored_img):
    print(colored_img[:,:,1])
    img = colored_img.copy()
    mask = get_bayer_masks(colored_img.shape[0], colored_img.shape[1])
    stepSize = 1
    windowSize = (3,3)
    for x in range(0, img.shape[0] - 2, stepSize):
        for y in range(0, img.shape[1] - 2, stepSize):
            window = colored_img[x:x + windowSize[0], y:y + windowSize[1],:]
            mask_window = mask[x:x + windowSize[0], y:y + windowSize[1],:].astype('uint8')
            nz0 = (((x+1) % 2) + 1) * (y % 2 + 1)
            nz1 = ((x + y) % 2) * 4 + ((x + y + 1) % 2) * 5
            nz2 = ((x % 2) + 1) * ((y+1) % 2 + 1)
            if mask_window[1,1,0] == 0:
                if nz0 == 2 and x % 2 == 0:
                    img[x+1,y+1,0] = (int(colored_img[x,y+1,0]) + colored_img[x+2,y+1,0]) // nz0
                if nz0 == 2 and x % 2 == 1:
                    img[x+1,y+1,0] = (int(colored_img[x+1,y+0,0]) + colored_img[x+1,y+2,0]) // nz0
                if nz0 == 4:
                    img[x+1,y+1,0] = (int(colored_img[x+0,y+0,0]) + colored_img[x+0,y+2,0] + colored_img[x+2,y+0,0] + colored_img[x+2,y+2,0]) // nz0
            if mask_window[1,1,1] == 0:
                if (nz1 == 4):
                    img[x+1,y+1,1] = (int(colored_img[x,y+1,1]) + colored_img[x+1,y,1] + colored_img[x+1,y+2,1] + colored_img[x+2,y+1,1]) // nz1
                if (nz1 == 5):
                    img[x+1,y+1,1] = (int(colored_img[x,y,1]) + colored_img[x+2,y,1] + colored_img[x,y+2,1] + colored_img[x+2,y+2,1] + colored_img[x+1,y+1,1]) // nz1
            if mask_window[1,1,2] == 0:
                if nz2 == 2 and x % 2 == 1:
                    img[x+1,y+1,2] = (int(colored_img[x,y+1,2]) + colored_img[x+2,y+1,2]) // nz2
                if nz2 == 2 and x % 2 == 0:
                    img[x+1,y+1,2] = (int(colored_img[x+1,y+0,2]) + colored_img[x+1,y+2,2]) // nz2
                if nz2 == 4:
                    img[x+1,y+1,2] = (int(colored_img[x+0,y+0,2]) + colored_img[x+0,y+2,2] + colored_img[x+2,y+0,2] + colored_img[x+2,y+2,2]) // nz2
    print(img[:,:,1])
    return img

def improved_interpolation(raw_image):
    h = raw_image.shape[0]
    w = raw_image.shape[1]
    raw_image = np.float64(raw_image)/255
    weigths_0 = (1 / 8) * np.array([
        [ 0,  0, 1/2,  0,  0],
        [ 0, -1,   0, -1,  0],
        [-1,  4,   5,  4, -1],
        [ 0, -1,   0, -1,  0],
        [ 0,  0, 1/2,  0,  0]
    ])

    weigths_1 = (1 / 8) * np.array([
        [  0,  0,  -1,  0,   0],
        [  0, -1,   4, -1,   0],
        [1/2,  0,   5,  0, 1/2],
        [  0, -1,   4, -1,   0],
        [  0,  0,  -1,  0,   0]
    ])

    weigths_2 = (1 / 8) * np.array([
        [   0,  0, -3/2,  0,    0],
        [   0,  2,    0,  2,    0],
        [-3/2,  0,    6,  0, -3/2],
        [   0,  2,    0,  2,    0],
        [   0,  0, -3/2,  0,    0]
    ])

    weigths_3 = (1 / 8) * np.array([
        [ 0,  0, -1,  0,  0],
        [ 0,  0,  2,  0,  0],
        [-1,  2,  4,  2, -1],
        [ 0,  0,  2,  0,  0],
        [ 0,  0, -1,  0,  0]
    ])
    image = np.zeros((h, w, 3), dtype=np.float64)
    mask_r, mask_g1, mask_g2, mask_b = get_bayer_pattern_masks(h, w)
    #print(mask_r)
    image[:,:,0] = convolve(raw_image, weigths_0) * mask_g2 + convolve(raw_image, weigths_1) * mask_g1 + convolve(raw_image, weigths_2) * mask_b + raw_image * mask_r
    image[:,:,1] = convolve(raw_image, weigths_3) * (mask_r + mask_b) + raw_image * (mask_g1+mask_g2)
    image[:,:,2] = convolve(raw_image, weigths_2) * (mask_r) + convolve(raw_image, weigths_0) * (mask_g1) + convolve(raw_image, weigths_1) * mask_g2 + raw_image * (mask_b)
    print(image.max())
    return (image * 255).astype('int32').clip(0, 255).astype('uint8')

def main():
    raw_img = np.array([[8, 5, 3, 7, 1, 3],
                        [5, 2, 6, 8, 8, 1],
                        [9, 9, 8, 1, 6, 4],
                        [9, 4, 2, 3, 6, 8],
                        [5, 4, 3, 2, 8, 7],
                        [7, 3, 3, 6, 9, 3]])
    print("img before")
    print(raw_img)
    gt_img = np.zeros((6, 6, 3), 'uint8')
    r = slice(2, -2), slice(2, -2)
    img = improved_interpolation(raw_img)
    gt_img[r + (0,)] = np.array([[6, 1],
                              [1, 0]])
    gt_img[r + (1,)] = np.array([[8, 4],
                              [2, 3]])
    gt_img[r + (2,)] = np.array([[7, 2],
                              [2, 2]])
    print("my img:")
    print(img[:,:,1])
    print("gt img part:")
    print(gt_img[:,:,1][r])
    print("my img part:")
    print(img[:,:,1][r])
    
def compute_psnr(img_pred,img_gt):
    compressed = img_pred.astype('float64')
    original = img_gt.astype('float64')
    mse = np.mean((original - compressed) ** 2)
    if(mse == 0):
        raise ValueError('mse = 0')
    max_pixel = img_gt.max()
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr
        
        
if __name__ == '__main__':
    main()
    

