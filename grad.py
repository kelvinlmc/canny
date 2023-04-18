import numpy as np
from gauss import conv

def partial_x(img):
    """ Computes partial x-derivative of input img.#计算输入图像x的偏导数

    Hints:
        - You may use the conv function in defined in this file.

    Args:
        img: numpy array of shape (H, W).
    Returns:
        out: x-derivative image.  #x的导数像
    """

    out = None

    ### YOUR CODE HERE
    # 对x求偏导
    kernel = np.array(([-1, 0, 1], [-1, 0, 1], [-1, 0, 1]))  # np.random.randint(10, size=(3, 3))
    out = conv(img, kernel) / 2
    ### END YOUR CODE

    return out


def partial_y(img):
    """ Computes partial y-derivative of input img.

    Hints:
        - You may use the conv function in defined in this file.

    Args:
        img: numpy array of shape (H, W).
    Returns:
        out: y-derivative image.
    """

    out = None

    ### YOUR CODE HERE
    # 对y求偏导
    kernel = np.array(([1, 1, 1], [0, 0, 0], [-1, -1, -1]))  # np.random.randint(10, size=(3, 3))
    out = conv(img, kernel) / 2
    ### END YOUR CODE

    return out


def gradient(img):
    """ Returns gradient magnitude and direction of input img. #返回图像的梯度大小和方向

    Args:
        img: Grayscale image. Numpy array of shape (H, W).灰度图像

    Returns:
        G: Magnitude of gradient at each pixel in img.  #返回每个像素的梯度大小
            Numpy array of shape (H, W).
        theta: Direction(in degrees, 0 <= theta < 360) of gradient  #方向0-360度
            at each pixel in img. Numpy array of shape (H, W).

    Hints:
        - Use np.sqrt and np.arctan2 to calculate square root and arctan
    """
    G = np.zeros(img.shape)
    theta = np.zeros(img.shape)

    ### YOUR CODE HERE
    Gx = partial_x(img)
    Gy = partial_y(img)
    # 求梯度的大小(平方和的根号)
    G = np.sqrt(np.power(Gx, 2) + np.power(Gy, 2))
    theta = np.arctan2(Gy, Gx) * 180 / np.pi  # 转换成角度制
    ### END YOUR CODE

    return G, theta

import numpy as np
import matplotlib.pyplot as plt
import torch
import cv2
import io

def gaussian_kernel(size, sigma):
    """ Implementation of Gaussian Kernel.

    This function follows the gaussian kernel formula,
    and creates a kernel matrix.

    Hints:
    - Use np.pi and np.exp to compute pi and exp.

    Args:
        size: int of the size of output matrix.
        sigma: float of sigma to calculate kernel.

    Returns:
        kernel: numpy array of shape (size, size).
        #高斯核的实现
        #这个函数遵循高斯核公式，并创建一个核矩阵
        #
    """

    kernel = np.zeros((size, size))

    ### YOUR CODE HERE
    for x in range(size):
        for y in range(size):
            kernel[x, y] = 1 / (2*np.pi*sigma*sigma) * np.exp(-(x*x+y*y)/(2*sigma*sigma))
    ### END YOUR CODE

    return kernel

def conv(image, kernel):
    """ An implementation of convolution filter.

    This function uses element-wise multiplication and np.sum()
    to efficiently compute weighted sum of neighborhood at each
    pixel.

    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk).

    Returns:
        out: numpy array of shape (Hi, Wi).
        #卷积率滤波器的实现
        #这个函数使用逐元素乘法和np.sum()来有效的计算每个像素的领域加权和
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    # For this assignment, we will use edge values to pad the images.
    # Zero padding will make derivatives at the image boundary very big,
    # whereas we want to ignore the edges at the boundary.
    pad_width0 = Hk // 2
    pad_width1 = Wk // 2
    pad_width = ((pad_width0,pad_width0),(pad_width1,pad_width1))
    padded = np.pad(image, pad_width, mode='edge')

    ### YOUR CODE HERE
    x = Hk // 2
    y = Wk // 2
    # 横向遍历卷积后的图像
    for i in range(pad_width0, Hi-pad_width0):
        # 纵向遍历卷积后的图像
        for j in range(pad_width1, Wi-pad_width1):
            split_img = image[i-pad_width0:i+pad_width0+1, j-pad_width1:j+pad_width1+1]
            # 对应元素相乘
            out[i, j] = np.sum(np.multiply(split_img, kernel)) #np.multiply是数组对应元素相乘
    # out = (out-out.min()) * (1/(out.max()-out.min()) * 255).astype('uint8')
    ### END YOUR CODE

    return out
if __name__ =='__main__':
    # 用自己写的高斯核滤波卷积核去卷积图片
    # Test with different kernel_size and sigma
    kernel_size = 5
    sigma = 1.4

    # Load image
    img = cv2.imread('./xi.jpg', cv2.IMREAD_GRAYSCALE)
    kernel = gaussian_kernel(kernel_size, sigma)

    # Convolve image with kernel to achieve smoothed effect
    smoothed = conv(img, kernel)

    # Compute partial derivatives of smoothed image
    Gx = partial_x(smoothed)
    Gy = partial_y(smoothed)
    yuanx = partial_x(img)
    yuany = partial_y(img)

    plt.subplot(1, 4, 1)
    plt.imshow(Gx, cmap="gray")
    plt.title('Derivative in x direction')
    plt.axis('off')

    plt.subplot(1, 4, 2)
    plt.imshow(Gy, cmap="gray")
    plt.title('Derivative in y direction')
    plt.axis('off')

    plt.subplot(2, 4, 3)
    plt.imshow(yuanx, cmap="gray")
    plt.title('yuan in x direction')
    plt.axis('off')

    plt.subplot(2, 4, 4)
    plt.imshow(yuany, cmap="gray")
    plt.title('yuan in y direction')
    plt.axis('off')

    plt.show()


