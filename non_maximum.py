import numpy as np
from gauss import conv

#测试使用库
import cv2
import matplotlib.pyplot as plt
from gauss import gaussian_kernel,conv
from grad import gradient

def non_maximum_suppression(G, theta):
    """ Performs non-maximum suppression.  #执行非极大值抑制

    This function performs non-maximum suppression along the direction
    of gradient (theta) on the gradient magnitude image (G).沿着梯度方向对梯度值进行非极大值抑制

    Args:
        G: gradient magnitude image with shape of (H, W).
        theta: direction of gradients with shape of (H, W).

    Returns:
        out: non-maxima suppressed image.
    """
    H, W = G.shape
    out = np.zeros((H, W))

    # 将角度近似到45°的对角线, 正负无所谓, 只看对角线跟水平垂直
    # Round the gradient direction to the nearest 45 degrees
    theta = np.floor((theta + 22.5) / 45) * 45

    ### BEGIN YOUR CODE
    anchor = np.stack(np.where(G != 0)).T  # 获取非零梯度的位置
    for x, y in anchor:
        center_point = G[x, y]
        current_theta = theta[x, y]
        dTmp1 = 0
        dTmp2 = 0
        W = 0
        if current_theta >= 0 and current_theta < 45:
            g1, g2, g3, g4 = G[x + 1, y - 1], G[x + 1, y], G[x - 1, y + 1], G[x - 1, y]
            W = abs(np.tan(current_theta * np.pi / 180))  # tan0-45范围为0-1
            dTmp1 = W * g1 + (1 - W) * g2
            dTmp2 = W * g3 + (1 - W) * g4
        elif current_theta >= 45 and current_theta < 90:
            g1, g2, g3, g4 = G[x + 1, y - 1], G[x, y - 1], G[x - 1, y + 1], G[x, y + 1]
            W = abs(np.tan((current_theta - 90) * np.pi / 180))
            dTmp1 = W * g1 + (1 - W) * g2
            dTmp2 = W * g3 + (1 - W) * g4
        elif current_theta >= -90 and current_theta < -45:
            g1, g2, g3, g4 = G[x - 1, y - 1], G[x, y - 1], G[x + 1, y + 1], G[x, y + 1]
            W = abs(np.tan((current_theta - 90) * np.pi / 180))
            dTmp1 = W * g1 + (1 - W) * g2
            dTmp2 = W * g3 + (1 - W) * g4
        elif current_theta >= -45 and current_theta < 0:
            g1, g2, g3, g4 = G[x + 1, y + 1], G[x + 1, y], G[x - 1, y - 1], G[x - 1, y]
            W = abs(np.tan(current_theta * np.pi / 180))
            dTmp1 = W * g1 + (1 - W) * g2
            dTmp2 = W * g3 + (1 - W) * g4
        if dTmp1 < center_point and dTmp2 < center_point:
            out[x, y] = center_point
    return out
if __name__ =='__main__':

    #
    kernel_size = 5
    sigma = 1.4
    img = cv2.imread('./vgg_data/2007_000170.jpg', cv2.IMREAD_GRAYSCALE)


    # 高斯模糊
    kernel = gaussian_kernel(kernel_size, sigma)
    img1 = conv(img, kernel)
    # 计算梯度Prewitt算子
    G, theta = gradient(img1)
    # io.imshow(color.gray2rgb(G))
    # 非极大值抑制
    G = non_maximum_suppression(G, theta)

    plt.subplot(1, 3, 1)
    plt.imshow(img, cmap="gray")
    plt.title('Original image')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(img1, cmap="gray")
    plt.title('smoothed')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(G, cmap="gray")
    plt.title('non_maximum')
    plt.axis('off')

    plt.show()
