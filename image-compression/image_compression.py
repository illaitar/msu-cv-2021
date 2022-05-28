import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from scipy.ndimage.filters import gaussian_filter
from skimage.metrics import peak_signal_noise_ratio
from numpy import linalg as LA
import scipy.ndimage as ndimage
import math
# !Этих импортов достаточно для решения данного задания, нельзя использовать другие библиотеки!


def pca_compression(matrix, p):
    """ Сжатие изображения с помощью PCA
    Вход: двумерная матрица (одна цветовая компонента картинки), количество компонент
    Выход: собственные векторы и проекция матрицы на новое пр-во
    """
    # Отцентруем каждую строчку матрицы
    matrix = np.float64(matrix)
    M_mean = np.mean(matrix,axis = 1)
    for i in range(matrix.shape[0]):
        matrix[i,:] -= M_mean[i]
            
    # Найдем матрицу ковариации
    COV = np.cov(matrix)
    # Ищем собственные значения и собственные векторы матрицы ковариации, используйте linalg.eigh из numpy
    eig_val, eig_vec = LA.eigh(COV)
    # Посчитаем количество найденных собственных векторов
    count = eig_vec.shape[1]
    
    # Сортируем собственные значения в порядке убывания
    val_sorted = np.argsort(eig_val)[::-1]
    
    # Сортируем собственные векторы согласно отсортированным собственным значениям
    # !Это все для того, чтобы мы производили проекцию в направлении максимальной дисперсии!
    eig_vec = eig_vec[:,val_sorted]
    # Оставляем только p собственных векторов
    eig_vec = eig_vec[:,:p]
    # Проекция данных на новое пространство
    res = np.dot(eig_vec.T, matrix)
    return eig_vec, res, M_mean


def pca_decompression(compressed):
    """ Разжатие изображения
    Вход: список кортежей из собственных векторов и проекций для каждой цветовой компоненты
    Выход: разжатое изображение
    """
    result_img = []
    for i, comp in enumerate(compressed):
        vectors = comp[0].copy()
        proj = comp[1].copy()
        mean = compressed[0][2].copy()
        # Матрично умножаем собственные векторы на проекции и прибавляем среднее значение по строкам исходной матрицы
        # !Это следует из описанного в самом начале примера!
        layer = np.dot(vectors, proj)
        for i in range(layer.shape[0]):
            layer[i,:] += mean[i]
        result_img.append(layer)
    res = np.stack(np.clip(np.array(result_img).astype('int32'),0,255).astype('uint8'),axis=2)
    return res


def pca_visualize():
    plt.clf()
    img = imread('cat.jpg')
    if len(img.shape) == 3:
        img = img[..., :3]
    fig, axes = plt.subplots(3, 3)
    fig.set_figwidth(12)
    fig.set_figheight(12)

    for i, p in enumerate([1, 5, 10, 20, 50, 100, 150, 200, 256]):
        compressed = []
        for j in range(0, 3):
            compressed.append(pca_compression(img[:,:,j],p))
        compressed = pca_decompression(compressed)
        axes[i // 3, i % 3].imshow(compressed)
        axes[i // 3, i % 3].set_title('Компонент: {}'.format(p))

    fig.savefig("pca_visualization.png")


def rgb2ycbcr(img):
    """ Переход из пр-ва RGB в пр-во YCbCr
    Вход: RGB изображение
    Выход: YCbCr изображение
    """
    r,g,b = [img[:,:,i] for i in range(3)]
    Y = 0.299 * r + 0.587 * g + 0.114 * b
    Cb = 128 - 0.1687 * r - 0.3313 * g + 0.5 * b
    Cr = 128 + 0.5 * r - 0.4187 * g - 0.0813 * b
    return np.clip(np.dstack((Y,Cb,Cr)),0,255)


def ycbcr2rgb(img):
    """ Переход из пр-ва YCbCr в пр-во RGB
    Вход: YCbCr изображение
    Выход: RGB изображение
    """
    Y,Cb,Cr = [img[:,:,i] for i in range(3)]
    r = Y + 0.1402 * (Cr-128)
    g = Y - 0.34414 * (Cb-128) - 0.71414 * (Cr-128)
    b = Y + 1.77 * (Cb-128)
    return np.clip(np.dstack((r,g,b)),0,255)


def get_gauss_1():
    plt.clf()
    rgb_img = imread('Lenna.png')
    if len(rgb_img.shape) == 3:
        rgb_img = rgb_img[..., :3]
    y = rgb2ycbcr(rgb_img)
    y[:,:,1] = ndimage.filters.gaussian_filter(y[:,:,1], sigma=10)
    y[:,:,2] = ndimage.filters.gaussian_filter(y[:,:,2], sigma=10)
    res = np.clip(ycbcr2rgb(y).astype('int'),0,255)
    plt.imshow(res)
    plt.savefig("gauss_1.png")


def get_gauss_2():
    plt.clf()
    rgb_img = imread('Lenna.png')
    if len(rgb_img.shape) == 3:
        rgb_img = rgb_img[..., :3]
    y = rgb2ycbcr(rgb_img)
    y[:,:,0] = ndimage.filters.gaussian_filter(y[:,:,0], sigma=10)
    res = np.clip(ycbcr2rgb(y).astype('int'),0,255)
    plt.imshow(res)
    plt.savefig("gauss_2.png")


def downsampling(component):
    """Уменьшаем цветовые компоненты в 2 раза
    Вход: цветовая компонента размера [A, B, 1]
    Выход: цветовая компонента размера [A // 2, B // 2, 1]
    """
    
    # Your code here
    comp = ndimage.filters.gaussian_filter(component, sigma=10.0)
    res = comp[::2,::2]
    return res


def dct(block):
    """Дискретное косинусное преобразование
    Вход: блок размера 8x8
    Выход: блок размера 8x8 после ДКП
    """
    res = np.zeros((8,8))
    def alpha(u):
        if u == 0:
            return 1/math.sqrt(2)
        return 1
    for u in range(8):
        for v in range(8):
            sum = 0
            for x in range(8):
                for y in range(8):
                    sum += block[x,y] * math.cos((2*x+1) * u * math.pi / 16) * math.cos((2*y+1) * v * math.pi / 16)
            res[u,v] = 1/4 * alpha(u) * alpha(v) * sum
    return res


# Матрица квантования яркости
y_quantization_matrix = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
])

# Матрица квантования цвета
color_quantization_matrix = np.array([
    [17, 18, 24, 47, 99, 99, 99, 99],
    [18, 21, 26, 66, 99, 99, 99, 99],
    [24, 26, 56, 99, 99, 99, 99, 99],
    [47, 66, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99]
])


def quantization(block, quantization_matrix):
    """Квантование
    Вход: блок размера 8x8 после применения ДКП; матрица квантования
    Выход: блок размера 8x8 после квантования. Округление осуществляем с помощью np.round
    """
    res = np.round(block / quantization_matrix)
    return res


def own_quantization_matrix(default_quantization_matrix, q):
    """Генерация матрицы квантования по Quality Factor
    Вход: "стандартная" матрица квантования; Quality Factor
    Выход: новая матрица квантования
    Hint: если после проделанных операций какие-то элементы обнулились, то замените их единицами
    """

    assert 1 <= q <= 100
    s = 1
    if 1 <= q < 50:
        s = 5000/q
    if 50 <= q <= 99:
        s = 200 - 2 * q
    res = np.zeros(default_quantization_matrix.shape)
    for i in range(default_quantization_matrix.shape[0]):
        for k in range(default_quantization_matrix.shape[1]):
            res[i,k] = np.floor((50 + s * default_quantization_matrix[i,k]) / 100)
            if res[i,k] == 0:
                res[i,k] = 1
    return res


def zigzag(block):
    """Зигзаг-сканирование
    Вход: блок размера 8x8
    Выход: список из элементов входного блока, получаемый после его обхода зигзаг-сканированием
    """
    zigzag = []
    z = np.flip(block,axis=1)
    len = block.shape[0] - 1
    for k in range(-len,len+1):
        i = -k
        diag = np.diagonal(z,offset = i)
        if abs(k % 2) == 1:
            diag = diag[::-1]
        diag = np.array(diag)
        for l in range(diag.shape[0]):
            zigzag.append(diag[l])
    return zigzag


def compression(zigzag_list):
    """Сжатие последовательности после зигзаг-сканирования
    Вход: список после зигзаг-сканирования
    Выход: сжатый список в формате, который был приведен в качестве примера в самом начале данного пункта
    """

    # Your code here
    zigzag = zigzag_list
    zigzag2 = []
    flag = False
    num = 0
    for i in range(len(zigzag)):
        if flag:
            if zigzag[i] == 0:
                num+=1
            else:
                flag = False
                zigzag2.append(num)
                zigzag2.append(zigzag[i])
        else:
            if zigzag[i] == 0:
                num = 1
                flag = True
                zigzag2.append(zigzag[i])
            else:
                zigzag2.append(zigzag[i])
    if flag:
        zigzag2.append(num)
    return zigzag2


def jpeg_compression(img, quantization_matrixes):
    """JPEG-сжатие
    Вход: цветная картинка, список из 2-ух матриц квантования
    Выход: список списков со сжатыми векторами: [[compressed_y1,...], [compressed_Cb1,...], [compressed_Cr1,...]]
    """

    # Your code here
        
    # Переходим из RGB в YCbCr
    y = rgb2ycbcr(img)
    y,b,r = [y[:,:,i] for i in range(3)]
    # Уменьшаем цветовые компоненты
    ds_y = y
    ds_b = downsampling(b)
    ds_r = downsampling(r)
    
    # Делим все компоненты на блоки 8x8 и все элементы блоков переводим из [0, 255] в [-128, 127]
    x = ds_y.shape[0] // 8
    y = ds_y.shape[1] // 8
    def get_blocks(img,x,y):
        blocks = []
        for i in range(x):
            for k in range(y):
                blocks.append(img[i * 8: (i+1) * 8,k * 8: (k+1) * 8] - 128)
        return blocks
    blocks_y = get_blocks(ds_y, x, y)
    x = ds_b.shape[0] // 8
    y = ds_b.shape[1] // 8
    blocks_b = get_blocks(ds_b, x, y)
    blocks_r = get_blocks(ds_r, x, y)
    # Применяем ДКП, квантование, зизгаз-сканирование и сжатие
    comp_y = []
    for z in blocks_y:
        comp_y.append(compression(zigzag(quantization(dct(z),quantization_matrixes[0]))))
    comp_b = []
    for z in blocks_b:
        comp_b.append(compression(zigzag(quantization(dct(z),quantization_matrixes[1]))))
    comp_r = []
    for z in blocks_r:
        comp_r.append(compression(zigzag(quantization(dct(z),quantization_matrixes[1]))))
    return [comp_y,comp_b,comp_r]


def inverse_compression(compressed_list):
    """Разжатие последовательности
    Вход: сжатый список
    Выход: разжатый список
    """
    zigzag = []
    flag = False
    num = 0
    for i in range(len(compressed_list)):
        if not flag:
            if compressed_list[i] == 0:
                for i in range(compressed_list[i+1]):
                    zigzag.append(0.0)
                flag = True
            else:
                zigzag.append(compressed_list[i])
        else:
            flag = False
    return zigzag


def inverse_zigzag(input):
    """Обратное зигзаг-сканирование
    Вход: список элементов
    Выход: блок размера 8x8 из элементов входного списка, расставленных в матрице в порядке их следования в зигзаг-сканировании
    """
    lens = [1,2,3,4,5,6,7,8,7,6,5,4,3,2,1]
    k = 0
    matr = np.zeros((8,8))
    flag = False
    for i in lens:
        diag = np.array(input[k:k+i])
        if i%2 == 0:
            diag = diag[::-1]
        x_start = 0
        y_start = i - 1
        if i == 8:
            flag = True
        if flag:
            x_start = 8 - i
            y_start = 7
        first_coord = (y_start,x_start)
        for p in range(len(diag)):
            matr[y_start-p,x_start+p] = diag[p]
        k+=i
    return matr

def inverse_quantization(block, quantization_matrix):
    """Обратное квантование
    Вход: блок размера 8x8 после применения обратного зигзаг-сканирования; матрица квантования
    Выход: блок размера 8x8 после квантования. Округление не производится
    """
    res = block * quantization_matrix
    return res


def inverse_dct(block):
    """Обратное дискретное косинусное преобразование
    Вход: блок размера 8x8
    Выход: блок размера 8x8 после обратного ДКП. Округление осуществляем с помощью np.round
    """
    res = np.zeros((8,8))
    def alpha(u):
        if u == 0:
            return 1/math.sqrt(2)
        return 1
    for x in range(8):
        for y in range(8):
            sum = 0
            for u in range(8):
                for v in range(8):
                    sum += alpha(u) * alpha(v) * block[u,v] * math.cos((2*x+1) * u * math.pi / 16) * math.cos((2*y+1) * v * math.pi / 16)
            res[x,y] = np.round(1/4 * sum)
    return res


def upsampling(component):
    """Увеличиваем цветовые компоненты в 2 раза
    Вход: цветовая компонента размера [A, B, 1]
    Выход: цветовая компонента размера [2 * A, 2 * B, 1]
    """
    res = np.zeros((component.shape[0]*2,component.shape[1]*2))
    for i in range(component.shape[0]):
        for k in range(component.shape[1]):
            res[i*2,k*2] = component[i,k]
            res[i*2+1,k*2] = component[i,k]
            res[i*2,k*2+1] = component[i,k]
            res[i*2+1,k*2+1] = component[i,k]
    return res


def jpeg_decompression(result, result_shape, quantization_matrixes):
    """Разжатие изображения
    Вход: result список сжатых данных, размер ответа, список из 2-ух матриц квантования
    Выход: разжатое изображение
    """
    blocks1 = []
    for vec1 in result[0]:
        blocks1.append(inverse_dct(inverse_quantization(inverse_zigzag(inverse_compression(vec1)),quantization_matrixes[0])))
    res1 = np.zeros((result_shape[0],result_shape[1]))
    tmp = 0
    for i in range(result_shape[0]//8):
        for k in range(result_shape[1]//8):
            res1[i*8:(i+1)*8, k*8:(k+1)*8] = blocks1[tmp] + 128
            tmp+=1
            
            
    blocks2 = []
    for vec1 in result[1]:
        blocks2.append(inverse_dct(inverse_quantization(inverse_zigzag(inverse_compression(vec1)),quantization_matrixes[1])))
    res2 = np.zeros((result_shape[0]//2,result_shape[1]//2))
    tmp = 0
    for i in range(result_shape[0]//16):
        for k in range(result_shape[1]//16):
            res2[i*8:(i+1)*8, k*8:(k+1)*8] = blocks2[tmp] + 128
            tmp+=1
            
            
    blocks3 = []
    for vec1 in result[2]:
        blocks3.append(inverse_dct(inverse_quantization(inverse_zigzag(inverse_compression(vec1)),quantization_matrixes[1])))
    res3 = np.zeros((result_shape[0]//2,result_shape[1]//2))
    tmp = 0
    for i in range(result_shape[0]//16):
        for k in range(result_shape[1]//16):
            res3[i*8:(i+1)*8, k*8:(k+1)*8] = blocks3[tmp] + 128
            tmp+=1
    res2 = upsampling(res2)
    res3 = upsampling(res3)
    ybr = np.dstack((res1,res2,res3))
    print(ybr.shape)
    rgb_img = ycbcr2rgb(ybr)
    res = np.clip(np.array(rgb_img).astype('int32'),0,255).astype('uint8')
    return res


def jpeg_visualize():
    plt.clf()
    img = imread('Lenna.png')
    if len(img.shape) == 3:
        img = img[..., :3]
    fig, axes = plt.subplots(2, 3)
    fig.set_figwidth(12)
    fig.set_figheight(12)

    for i, p in enumerate([1, 10, 20, 50, 80, 100]):
        quantization_matrixes = [own_quantization_matrix(y_quantization_matrix,p),own_quantization_matrix(color_quantization_matrix,p)]
        compressed = jpeg_decompression(jpeg_compression(img,quantization_matrixes),img.shape,quantization_matrixes)
        axes[i // 3, i % 3].imshow(compressed)
        axes[i // 3, i % 3].set_title('Quality Factor: {}'.format(p))

    fig.savefig("jpeg_visualization.png")


def compression_pipeline(img, c_type, param=1):
    """Pipeline для PCA и JPEG
    Вход: исходное изображение; название метода - 'pca', 'jpeg'; 
    param - кол-во компонент в случае PCA, и Quality Factor для JPEG
    Выход: изображение; количество бит на пиксель
    """
    
    assert c_type.lower() == 'jpeg' or c_type.lower() == 'pca'
    
    if c_type.lower() == 'jpeg':
        y_quantization = own_quantization_matrix(y_quantization_matrix, param)
        color_quantization = own_quantization_matrix(color_quantization_matrix, param)
        matrixes = [y_quantization, color_quantization]
        
        compressed = jpeg_compression(img, matrixes)
        img = jpeg_decompression(compressed, img.shape, matrixes)
    elif c_type.lower() == 'pca':
        compressed = []
        for j in range(0, 3):
            compressed.append((pca_compression(img[:, :, j].astype(np.float64).copy(), param)))
            
        img = pca_decompression(compressed)
        compressed.extend([np.mean(img[:, :, 0], axis=1), np.mean(img[:, :, 1], axis=1), np.mean(img[:, :, 2], axis=1)])
        
    if 'tmp' not in os.listdir() or not os.path.isdir('tmp'):
        os.mkdir('tmp')
        
    np.savez_compressed(os.path.join('tmp', 'tmp.npz'), compressed)
    size = os.stat(os.path.join('tmp', 'tmp.npz')).st_size * 8
    os.remove(os.path.join('tmp', 'tmp.npz'))
        
    return img, size / (img.shape[0] * img.shape[1])


def calc_metrics(img_path, c_type, param_list):
    """Подсчет PSNR и Rate-Distortion для PCA и JPEG. Построение графиков
    Вход: пусть до изображения; тип сжатия; список параметров: кол-во компонент в случае PCA, и Quality Factor для JPEG
    """
    
    assert c_type.lower() == 'jpeg' or c_type.lower() == 'pca'
    
    img = imread(img_path)
    if len(img.shape) == 3:
        img = img[..., :3]
    
    outputs = []
    for param in param_list:
        outputs.append(compression_pipeline(img.copy(), c_type, param))
     
    psnr = [peak_signal_noise_ratio(img, output[0]) for output in outputs]
    rate = [output[1] for output in outputs]
    
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
    fig.set_figwidth(20)
    fig.set_figheight(5)
    
    ax1.set_title('PSNR for {}'.format(c_type.upper()))
    ax1.plot(param_list, psnr, 'tab:orange')
    ax1.set_xlabel('Quality Factor')
    ax1.set_ylabel('PSNR')
    
    ax2.set_title('Rate-Distortion for {}'.format(c_type.upper()))
    ax2.plot(psnr, rate, 'tab:red')
    ax2.set_xlabel('Distortion')
    ax2.set_ylabel('Rate')
    return fig


def get_pca_metrics_graph():
    plt.clf()
    fig = calc_metrics('Lenna.png', 'pca', [1, 5, 10, 20, 50, 100, 150, 200, 256])
    fig.savefig("pca_metrics_graph.png")


def get_jpeg_metrics_graph():
    plt.clf()
    fig = calc_metrics('Lenna.png', 'jpeg', [1, 10, 20, 50, 80, 100])
    fig.savefig("jpeg_metrics_graph.png")

if __name__ == "__main__":
    get_jpeg_metrics_graph()
