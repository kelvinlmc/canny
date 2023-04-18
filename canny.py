from gauss import gaussian_kernel,conv
from grad import gradient
from non_maximum import non_maximum_suppression
from double import double_thresholding
from get_neibhor import link_edges

def canny(img, kernel_size=5, sigma=1.4, high=20, low=15):
    """ Implement canny edge detector by calling functions above.

    Args:
        img: binary image of shape (H, W).
        kernel_size: int of size for kernel matrix.
        sigma: float for calculating kernel.
        high: high threshold for strong edges.
        low: low threashold for weak edges.
    Returns:
        edge: numpy array of shape(H, W).
    """
    ### YOUR CODE HERE
    # 高斯模糊
    kernel = gaussian_kernel(kernel_size, sigma)
    img = conv(img, kernel)
    #io.imshow(color.gray2rgb(img))
    # 计算梯度Prewitt算子
    G, theta = gradient(img)
    #io.imshow(color.gray2rgb(G))
    # 非极大值抑制
    G = non_maximum_suppression(G, theta)
    #io.imshow(color.gray2rgb(G))

    # 双阈值抑制
    strong_edges, weak_edges = double_thresholding(G, high, low)
    # 连通边
    edge = link_edges(strong_edges, weak_edges)
    ### END YOUR CODE

    return edge
