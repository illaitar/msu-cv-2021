import numpy as np
import math
from scipy.fftpack import fft, ifft,fft2,fftshift,ifft2
from math import log10, sqrt


def gaussian_kernel(size, sigma):
    """
    Построение ядра фильтра Гаусса.

    @param  size  int    размер фильтра (нечетный)
    @param  sigma float  параметр размытия
    @return numpy array  фильтр Гаусса размером size x size
    """
    kernel = np.zeros((size,size),dtype='float64')
    rad = size//2
    print(rad)
    print(sigma**2)
    for i in range(0,size):
        for k in range(0,size):
            r = math.sqrt((rad - i)**2 + (rad - k)**2)
            kernel[i,k] = 1/(2 * math.pi * (sigma**2)) * math.exp(-(r*r /(2 * (sigma**2))))
    kernel = kernel / np.sum(kernel)
    print(kernel)
    return kernel


def fourier_transform(h, shape):
    """
    Получение Фурье-образа искажающей функции

    @param  h            numpy array  искажающая функция h (ядро свертки)
    @param  shape        list         требуемый размер образа
    @return numpy array  H            Фурье-образ искажающей функции h
    """
    l = fft2(h,shape=shape)
    return l


def inverse_kernel(H, threshold=1e-10):
    """
    Получение H_inv

    @param  H            numpy array    Фурье-образ искажающей функции h
    @param  threshold    float          порог отсечения для избежания деления на 0
    @return numpy array  H_inv
    """
    H_inv = np.zeros(H.shape,dtype="complex")
    for i in range(H.shape[0]):
        for k in range(H.shape[1]):
            if (abs(H[i,k]) <= threshold):
                H_inv[i,k] = np.conj(0)
            else:
                H_inv[i,k] = 1/H[i,k]
    return H_inv


def inverse_filtering(blurred_img, h, threshold=1e-10):
    """
    Метод инверсной фильтрации

    @param  blurred_img    numpy array  искаженное изображение
    @param  h              numpy array  искажающая функция
    @param  threshold      float        параметр получения H_inv
    @return numpy array                 восстановленное изображение
    """
    G = fourier_transform(blurred_img,blurred_img.shape)
    H = fourier_transform(h,blurred_img.shape)
    H_inv = inverse_kernel(H,threshold)
    F = G * H_inv
    f = ifft2(F,blurred_img.shape)
    res = np.zeros(H.shape)
    for i in range(blurred_img.shape[0]):
        for k in range(blurred_img.shape[1]):
            res[i,k] = abs(f[i,k])
    return res


def wiener_filtering(blurred_img, h, K=0.001):
    """
    Винеровская фильтрация

    @param  blurred_img    numpy array  искаженное изображение
    @param  h              numpy array  искажающая функция
    @param  K              float        константа из выражения (8)
    @return numpy array                 восстановленное изображение
    """
    G = fourier_transform(blurred_img,blurred_img.shape)
    H = fourier_transform(h,blurred_img.shape)
    
    H_conj = np.conj(H)
    F = H_conj/(np.abs(np.square(H))+K)*G
    f = ifft2(F,blurred_img.shape)
    res = np.zeros(H.shape)
    for i in range(blurred_img.shape[0]):
        for k in range(blurred_img.shape[1]):
            res[i,k] = abs(f[i,k])
    return res


def compute_psnr(img1, img2):
    """
    PSNR metric

    @param  img1    numpy array   оригинальное изображение
    @param  img2    numpy array   искаженное изображение
    @return float   PSNR(img1, img2)
    """
    MSE = np.mean((img1-img2)**2)
    PSNR = 20 * log10(255/sqrt(MSE))
    return PSNR

if __name__== "__main__":
    fourier_transform(np.ones((3,3)),(3,3))
