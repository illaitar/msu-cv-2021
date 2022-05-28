import numpy as np
import matplotlib.pyplot as plt
from skimage import img_as_float
from skimage.transform import resize
import math
    
def channel_shift(ch, offset):
    h, w = ch.shape
    i, j = offset
    ch = np.roll(ch, (i, j), axis=(1, 0))
    hfrom, hto = max(j, 0), min(h, h + j)
    wfrom, wto = max(i, 0), min(w, w + i)
    return ch[hfrom:hto, wfrom:wto]
    
def image_shift(r,g,b, offset_r, offset_b):
    height, width = r.shape
    i, j = offset_r
    k, h = offset_b
    r = np.roll(r, (i, j), axis=(1, 0))
    b = np.roll(b, (k, h), axis=(1, 0))
    hfrom, hto = max(j, h, 0), min(height, height + j, height + h)
    wfrom, wto = max(i, k, 0), min(width, width + i, width + k)
    return np.dstack([z[hfrom:hto, wfrom:wto] for z in [r, g, b]])
    
def interval(ch, i, k):
    start_i = max(0, i)
    start_k = max(0, k)
    end_i = min(ch.shape[1], ch.shape[1] + i)
    end_k = min(ch.shape[0], ch.shape[0] + k)
    return [start_i, start_k, end_i, end_k]
    
def cross_correlation(ch1, ch2, i, k):
    start_x, start_y, end_x, end_y = interval(ch2, i, k)
    ch1_shifted = ch1[start_y : end_y, start_x : end_x]
    ch2_shifted = ch2[start_y - k:end_y - k, start_x - i:end_x - i]
    return np.sum(ch1_shifted * ch2_shifted)/ math.sqrt(np.sum(ch2_shifted**2) * np.sum(ch1_shifted**2))

def mse(ch1, ch2, i, k):
    start_x, start_y, end_x, end_y = interval(ch2, i, k)
    ch1_shifted = ch1[start_y : end_y, start_x : end_x]
    ch2_shifted = ch2[start_y - k:end_y - k, start_x - i:end_x - i]
    return np.sum(np.square(ch1_shifted - ch2_shifted)) / (end_x - start_x) * (end_y - start_y)
            
def best_offset(ch1, ch2, offset, metric = 'NCC'):
    #принимает на вход два канала и границы проверки
    if (metric == 'MSE'):
        h, w = ch1.shape
        best_mse = 10000000
        best_value = (0,0)
        for i in range(offset[0][0], offset[0][1]):
            for k in range(offset[1][0], offset[1][1]):
                ch_mse = mse(ch1, ch2,i,k)
                if ch_mse < best_mse:
                    best_mse = ch_mse
                    best_value = (i,k)
    else:
        h, w = ch1.shape
        best_mse = -1
        best_value = (0,0)
        for i in range(offset[0][0], offset[0][1]):
            for k in range(offset[1][0], offset[1][1]):
               
                ch_mse = cross_correlation(ch1, ch2,i,k)
                if ch_mse > best_mse:
                    best_mse = ch_mse
                    best_value = (i,k)
    return best_value
        
def pyramid_search(ch1, ch2, depth, verbose = False, offset_e = 7):
    # Получает на вход два канала
    h, w = ch1.shape
    offset = None
    while (depth != -1):
        power = 2 ** depth
        if not(power == 1):
            cur_ch1 = resize(ch1, ((h // power, w // power)))
            cur_ch2 = resize(ch2, ((h // power, w // power)))
        else:
            cur_ch1 = ch1
            cur_ch2 = ch2
        depth -= 1
        if offset is None:
            start_range = ((-offset_e,offset_e),(-20,20))
            offset = best_offset(cur_ch1, cur_ch2, start_range)
            pass
        else:
            offset = (offset[0] * 2, offset[1] * 2)
            new_range = ((offset[0] - offset_e, offset[0] + offset_e),(offset[1] - offset_e, offset[1] + offset_e))
            offset = best_offset(cur_ch1, cur_ch2, new_range)
            pass
    return offset
        
def align(cam, g_coord):
    cam = img_as_float(cam)
    h, w = cam.shape
    h //= 3
    h_cut = (h * 5) // 100
    w_cut = (w * 5)// 100
    b,g,r = [cam[i*h + h_cut:(i+1)*h-h_cut, w_cut: w-w_cut] for i in range(3)][:]
    if (h > 400):
        depth = 3
        offset_e = 3
    else:
        depth = 1
        offset_e = 15
    offset_r = pyramid_search(g, r, depth, False, offset_e)
    offset_b = pyramid_search(g, b, depth, False, offset_e)
    aligned_image = image_shift(r,g,b,offset_r,offset_b)
    b_row = g_coord[0] - offset_b[1] - h
    b_col = g_coord[1] - offset_b[0]
    r_row = g_coord[0] - offset_r[1] + h
    r_col = g_coord[1] - offset_r[0]
    return (aligned_image * 255).astype('int32').clip(0, 255).astype('uint8'), (b_row, b_col), (r_row, r_col)

