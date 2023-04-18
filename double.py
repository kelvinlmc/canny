import numpy as np
#测试使用库
import cv2
import matplotlib.pyplot as plt
from gauss import gaussian_kernel,conv
from grad import gradient
from non_maximum import non_maximum_suppression
from get_neibhor import link_edges
from torchvision.utils import save_image
import torch

def double_thresholding(img, high, low):
    """
    Args:
        img: numpy array of shape (H, W) representing NMS edge response.
        high: high threshold(float) for strong edges.
        low: low threshold(float) for weak edges.

    Returns:
        strong_edges: Boolean array representing strong edges.
            Strong edeges are the pixels with the values greater than
            the higher threshold.
        weak_edges: Boolean array representing weak edges.
            Weak edges are the pixels with the values smaller or equal to the
            higher threshold and greater than the lower threshold.
    """

    strong_edges = np.zeros(img.shape, dtype=np.bool)
    weak_edges = np.zeros(img.shape, dtype=np.bool)

    ### YOUR CODE HERE
    H, W = img.shape
    for i in range(0, H):
        for j in range(0, W):
            if img[i, j] > high:
                strong_edges[i, j] = True
            elif img[i, j] >= low:
                weak_edges[i, j] = True
    ### END YOUR CODE

    return strong_edges, weak_edges
if __name__ =='__main__':
    kernel_size = 5
    sigma = 1.4
    img = cv2.imread('./000070.png', cv2.IMREAD_GRAYSCALE)

    # 高斯模糊
    kernel = gaussian_kernel(kernel_size, sigma)
    img1 = conv(img, kernel)
    # 计算梯度Prewitt算子
    G, theta = gradient(img1)
    # io.imshow(color.gray2rgb(G))
    # 非极大值抑制
    G1 = non_maximum_suppression(G, theta)
    # 双阈值抑制
    strong_edges, weak_edges = double_thresholding(G1, high=20, low=15)
    edge = link_edges(strong_edges, weak_edges)
    #保存边缘图
    #cv2.imwrite('./xiufu/2007_000170_1.jpg',G)
    #G2=torch.tensor(G1)
    #save_image(G2, './xiufu/2007_000738_1.jpg')

    #plt.savefig('./xiufu/2007_000738_1.jpg')
    '''plt.imshow(G, cmap="gray")
    plt.title('graded')
    plt.axis('off')'''


    plt.subplot(2, 2, 1)
    plt.imshow(img, cmap="gray")
    plt.title('Original image')
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.imshow(G, cmap="gray")
    plt.title('graded')
    plt.axis('off')

    plt.subplot(2, 2, 3)
    plt.imshow(G1, cmap="gray")
    plt.title('non_maximum')
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.imshow(edge, cmap="gray")
    plt.title('result')
    plt.axis('off')
    plt.show()