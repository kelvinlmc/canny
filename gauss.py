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
    img = cv2.imread('./img.png', cv2.IMREAD_GRAYSCALE)
    # img = cv2.imread('./mao.3.jpg',0)
    # img = io.imread('mao.3.png', as_gray=True)
    # plt.imshow(img_gray,cmap="gray")
    print(img)
    # b,g,r = cv2.split(img)
    # rgb_image = cv2.merge([r,g,b])
    # Define 5x5 Gaussian kernel with std = sigma
    kernel = gaussian_kernel(kernel_size, sigma)

    # Convolve image with kernel to achieve smoothed effect
    smoothed = conv(img, kernel)

    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap="gray")
    plt.title('Original image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(smoothed, cmap="gray")
    plt.title('Smoothed image')
    plt.axis('off')

    plt.show()
