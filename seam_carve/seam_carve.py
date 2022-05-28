from scipy.ndimage import convolve
from skimage.io import imread
from math import sqrt
import matplotlib.pyplot as plt
import numpy as np

def yuv(rgb_img):
    r,g,b = [rgb_img[:,:,i] for i in range(3)]
    res = 0.299 * r + 0.587 * g + 0.114 * b
    return res
    
def gradmap(y):
    w_x = np.zeros((3,3),dtype = 'float64')
    w_y = np.zeros((3,3),dtype = 'float64')
    w_x[1,0] = -1
    w_x[1,2] = 1
    w_y[0,1] = -1
    w_y[2,1] = 1
    map_x = convolve(y, w_x, mode="nearest")
    map_y = convolve(y, w_y, mode="nearest")
    return np.sqrt(map_x**2 + map_y**2)

def emap(map):
    x,y = map.shape
    res = np.zeros((x,y),dtype='float64')
    res[0,:] = map[0,:]
    res_p = np.pad(res,(1,1),constant_values=1e20)
    map_p = np.pad(map,(1,1),constant_values=1e20)
    for i in range(2,x+1):
        for k in range(1,y+1):
            res_p[i,k] = min(res_p[i-1,k-1],res_p[i-1,k],res_p[i-1,k+1]) + map[i-1,k-1]
    return res_p[1:x+1,1:y+1]
    
def worstseam(map):
    map_p = np.pad(map,((0,0),(1,1)),constant_values=1e20)
    carve = []
    carve.append((map.shape[0] - 1, map[-1].argmin()))
#    print(carve)
    for k in range(0,map.shape[0]-1):
        i = map.shape[0] - k - 2
        last_pos = carve[-1][1]
#        print(np.argmin(map_p[i, last_pos-1:last_pos+2]))
#        print(map_p[i, last_pos-1:last_pos+2])
        new_pos = last_pos + np.argmin(map_p[i, last_pos:last_pos+3]) -1
        carve.append((i,new_pos))
#        print(carve[-1])
    return carve
    
def vmap(shape, carve):
    res = np.zeros(shape,dtype='uint8')
    for i in carve:
        res[i[0],i[1]] = 1
    return res
    
def get_seam(y,mask):
    g_map = gradmap(y)
    g_max = g_map.shape[0] * g_map.shape[1] * 256
    if mask is not None:
        mask = g_max * mask.astype(np.float64)
        g_map += mask
    e_map = emap(g_map)
    seam = worstseam(e_map)
    v_map = vmap(e_map.shape, seam) #+ g_map
    return v_map, seam
    
def shrink(im,carve):
    img = np.array(im)
    new_img = np.zeros((img.shape[0], img.shape[1] - 1,3))
    carve = np.array(carve)
    for i in range(img.shape[0]):
        new_img[:,:carve[i][1],:] = img[:,:carve[i][1],:]
        new_img[:,carve[i][1]:,:] = img[:,carve[i][1] +1:,:]
    return np.clip(new_img/255,0,1)

def shrink_mask(im,carve):
    img = np.array(im)
    new_img = np.zeros((img.shape[0], img.shape[1] - 1))
    carve = np.array(carve)
    for i in range(img.shape[0]):
        new_img[:,:carve[i][1]] = img[:,:carve[i][1]]
        new_img[:,carve[i][1]:] = img[:,carve[i][1] +1:]
    return new_img
    
def seam_carve(img, orientation, mask):
    if orientation == 'vertical shrink':
        y = yuv(img).T
        if mask is not None:
            mask = np.array(mask).T
        res, seam = get_seam(y,mask)
        ch_img = shrink(img,seam)
        if mask is not None:
            ch_mask = shrink_mask(mask,seam)
            ch_mask = ch_mask.T
        else:
            ch_mask = None
        return ch_img.T, ch_mask, res.T
    if orientation == 'horizontal shrink':
        y = yuv(img)
        if mask is not None:
            mask = np.array(mask)
        res, seam = get_seam(y,mask)
        ch_img = shrink(img,seam)
        if mask is not None:
            ch_mask = shrink_mask(mask,seam)
        else:
            ch_mask = None
        return ch_img, ch_mask, res
    if orientation == 'vertical expand':
        y = yuv(img).T
        if mask is not None:
            mask = np.array(mask).T
        res, seam = get_seam(y,mask)
        return img, img,res.T
    if orientation == 'horizontal expand':
        y = yuv(img)
        if mask is not None:
            mask = np.array(mask)
        res, seam = get_seam(y,mask)
        return img, img, res
    
    
if __name__ == '__main__':
#    img = imread('public_tests/01_test_img_input/' +  'img.png')
#    #img 0-255
#    #plt.imshow(img)
    matr = np.array([[2,3,3],[2,2,0],[1,1,1]])
#    en_map = emap(gradmap(yuv(img)))
#    l = worstseam(en_map)
#    g = vmap(en_map.shape,l)
#    g += np.clip(gradmap(yuv(img)).astype('uint8'),0,255)
#    plt.imshow(g)
##    plt.show()
#    print(np.pad(matr,(1,1),constant_values=10000))
#    print(emap(matr))
#    print(np.pad(matr,(1,1),constant_values=10000))
    print(emap(matr))
    print(vmap(matr.shape,worstseam(emap(matr))))
    pass
