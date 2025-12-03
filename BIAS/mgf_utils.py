import matplotlib.patches
import numpy as np
# import cupy as np
import matplotlib.pyplot as plt
import matplotlib
import cv2
import time
from tqdm import trange

def get_min_sigma(image):
    return max(2,image.shape[0]*0.02)


def generate_gaussian_distribution(m, n, mu_x, mu_y, sigma_x, sigma_y, rho):
    if sigma_x <= 0 or sigma_y <= 0:  
        raise ValueError("Sigma values must be greater than zero.")
    if rho < -1 or rho > 1: 
        raise ValueError("Rho value must be between -1 and 1.")
    x = np.linspace(0, n - 1, n)
    y = np.linspace(0, m - 1, m)
    X, Y = np.meshgrid(x, y) 
    _1_sigma_1 = 1/sigma_x
    _1_sigma_2 = 1/sigma_y
    X_centered = (X - mu_x) * _1_sigma_1
    Y_centered = (Y - mu_y) * _1_sigma_2
    Q_mat = X_centered**2 + Y_centered**2 - 2 * rho * X_centered * Y_centered
    Z = (1 / (2 * np.pi * sigma_x * sigma_y * np.sqrt(1 - rho**2))) * np.exp(-Q_mat / (2 * (1 - rho**2)))
    return Z, Q_mat, X_centered, Y_centered

def convolution(image, kernel):
    # we hypothesis that the image size is the same as the kernel size, which is the m * n.
    # which is pre_calculated in the former steps.
    # we are here to calculate the convolution of the image and the kernel.
    conv_rst = image * kernel
    return np.sum(conv_rst), conv_rst

def partial_derivatives(Q_mat,conv_rst, X_centered, Y_centered,mu_x, mu_y, sigma_x, sigma_y, rho):
    lambda_sigma = get_min_sigma(Q_mat)/(np.sqrt(Q_mat.shape[0])*0.7)# add a penalty to avoid the sigma to be too small?
    _1_sigma_1 = 1/sigma_x
    _1_sigma_2 = 1/sigma_y
    _1_1_rho_2 = 1/(1-rho**2)
    cross_term = rho * X_centered * Y_centered
    par_sigma_1 = np.sum(conv_rst * (X_centered**2 - rho * cross_term - 1))*_1_sigma_1 + lambda_sigma * _1_sigma_1 #+  0.001*sigma_x**2
    # should be *_1_sigma_1, but we try to add a + sigma to keep the sigma not be too small
    par_sigma_2 = np.sum(conv_rst * (Y_centered**2 - rho * cross_term - 1))*_1_sigma_2 + lambda_sigma * _1_sigma_2 #+  0.001*sigma_y**2
    par_mu_x = np.sum(conv_rst * (X_centered - rho * Y_centered))*_1_1_rho_2 * _1_sigma_1
    par_mu_y = np.sum(conv_rst * (Y_centered - rho * X_centered))*_1_1_rho_2 * _1_sigma_2
    par_rho = np.sum(conv_rst * (rho + (X_centered * Y_centered - rho * Q_mat)*_1_1_rho_2)) * _1_1_rho_2
    return par_sigma_1, par_sigma_2, par_mu_x, par_mu_y, par_rho
def single_iteration(img, row, col, x_0, y_0, sigma_1_0, sigma_2_0, rho_0,max_iter_time = 15):
    
    for _ in range(max_iter_time):
        target_z, target_q, X_centered, Y_centered = generate_gaussian_distribution(row, col, x_0, y_0, sigma_1_0, sigma_2_0, rho_0)
        sum_conv, conv_rst = convolution(img, target_z)
        #print(f'current value is {sum_conv}')
        #t1 = time.time()
        position_step = 0.1
        sigma_step = 4
        par_sigma_1, par_sigma_2, par_mu_x, par_mu_y, par_rho = partial_derivatives(target_q,conv_rst, X_centered, Y_centered,x_0, y_0, sigma_1_0, sigma_2_0, rho_0)
        # print(f"The partial derivatives are {par_sigma_1}, {par_sigma_2}, {par_mu_x}, {par_mu_y}, {par_rho}")
        x_0 += position_step * par_mu_x
        y_0 += position_step * par_mu_y
        sigma_1_0 += sigma_1_0 * sigma_step * par_sigma_1
        sigma_1_0 = max(sigma_1_0, 2)
        sigma_2_0 += sigma_2_0 * sigma_step * par_sigma_2
        sigma_2_0 = max(sigma_2_0, 2)
        rho_0 += par_rho * 0.05
        rho_0 = max(min(rho_0, 0.999), -0.999)
        # fig,axs = plt.subplots(1,3)
        # axs[0].imshow(img)
        # axs[0].set_title("original image")
        # axs[1].imshow(conv_rst)
        # axs[1].set_title("convolution result")
        # axs[2].imshow(target_z)
        # axs[2].set_title("target gaussian")
        # plt.show()
    return x_0, y_0, sigma_1_0, sigma_2_0, rho_0

def loop_for_multiple_gaussian_fitting(ori_img, max_peaks = 15,thres = 0.1):
    '''
    loop for max_peaks times to find the peaks in the image. However, the height of max peak must be larger than thres. \n
    return: a list of peaks, each peak is a list of [x, y, sigma_1, sigma_2, rho, max_img_val]
    '''
    peaks = []
    img = ori_img.copy() / (np.max(ori_img)+1e-5)
    img = img - np.mean(img)
    row, col = img.shape
    for _ in range(max_peaks):
        x_0 = np.argmax(img)%col
        y_0 = np.argmax(img)//col
        sigma_1_0 = get_min_sigma(img)
        sigma_2_0 = get_min_sigma(img)
        rho_0 = 0
        x_0, y_0, sigma_1_0, sigma_2_0, rho_0 = single_iteration(img, row, col, x_0, y_0, sigma_1_0, sigma_2_0, rho_0)
        tgt_z = generate_gaussian_distribution(row, col, x_0, y_0, sigma_1_0, sigma_2_0, rho_0)[0]
        conv_rst = convolution(img, tgt_z/np.max(tgt_z))[1]
        max_img_val = img.max()
        peaks.append([x_0, y_0, sigma_1_0, sigma_2_0, rho_0, max_img_val])
        img = img - conv_rst
        if  img.max() < thres:
            break
    return peaks

def loop_for_multiple_gaussian_fitting2(ori_img, max_peaks = 15,thres = 0.25):
    '''
    loop for max_peaks times to find the peaks in the image. However, the height of max peak must be larger than thres. \n
    return: a list of peaks, each peak is a list of [x, y, sigma_1, sigma_2, rho, max_img_val]
    '''
    peaks = []
    img = ori_img.copy() / (np.max(ori_img)+1e-5)
    img = img - np.mean(img)
    row, col = img.shape
    fit_rst = np.zeros_like(img)
    for _ in range(max_peaks):
        x_0 = np.argmax(img)%col
        y_0 = np.argmax(img)//col
        sigma_1_0 = get_min_sigma(img)
        sigma_2_0 = get_min_sigma(img)
        rho_0 = 0
        x_0, y_0, sigma_1_0, sigma_2_0, rho_0 = single_iteration(img, row, col, x_0, y_0, sigma_1_0, sigma_2_0, rho_0)
        tgt_z = generate_gaussian_distribution(row, col, x_0, y_0, sigma_1_0, sigma_2_0, rho_0)[0]
        conv_rst = convolution(img, tgt_z/np.max(tgt_z))[1]
        max_img_val = img.max()
        peaks.append([x_0, y_0, sigma_1_0, sigma_2_0, rho_0, max_img_val])
        img = img - conv_rst
        fit_rst += tgt_z
        if  img.max() < thres:
            break
    return fit_rst
    
def generate_test_image(shape=(100, 100)):
    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    # 单峰高斯分布
    gauss = 100 * np.exp(-((x-30)**2/(2*10**2) + (y-40)**2/(2*15**2)))
    return gauss

def numerical_gradient(param_name, epsilon=1e-5):
    # 初始参数
    mu_x, mu_y = 30.0, 40.0
    sigma_x, sigma_y = 10.0, 15.0
    rho = 0.0
    
    # 生成原始高斯分布和卷积结果
    Z_orig, Q_orig, Xc_orig, Yc_orig = generate_gaussian_distribution(
        row, col, mu_x, mu_y, sigma_x, sigma_y, rho
    )
    _, conv_rst_orig = convolution(test_img, Z_orig)
    
    # 扰动参数 +epsilon
    if param_name == "sigma_x":
        sigma_x_p = sigma_x + epsilon
        Z_p, _, _, _ = generate_gaussian_distribution(row, col, mu_x, mu_y, sigma_x_p, sigma_y, rho)
    elif param_name == "sigma_y":
        sigma_y_p = sigma_y + epsilon
        Z_p, _, _, _ = generate_gaussian_distribution(row, col, mu_x, mu_y, sigma_x, sigma_y_p, rho)
    elif param_name == "mu_x":
        mu_x_p = mu_x + epsilon
        Z_p, _, _, _ = generate_gaussian_distribution(row, col, mu_x_p, mu_y, sigma_x, sigma_y, rho)
    elif param_name == "mu_y":
        mu_y_p = mu_y + epsilon
        Z_p, _, _, _ = generate_gaussian_distribution(row, col, mu_x, mu_y_p, sigma_x, sigma_y, rho)
    elif param_name == "rho":
        rho_p = rho + epsilon
        Z_p, _, _, _ = generate_gaussian_distribution(row, col, mu_x, mu_y, sigma_x, sigma_y, rho_p)
    else:
        raise ValueError("Invalid parameter name")
    loss_p = np.sum(convolution(test_img, Z_p)[1])
    
    # 扰动参数 -epsilon
    if param_name == "sigma_x":
        sigma_x_m = sigma_x - epsilon
        Z_m, _, _, _ = generate_gaussian_distribution(row, col, mu_x, mu_y, sigma_x_m, sigma_y, rho)
    elif param_name == "sigma_y":
        sigma_y_m = sigma_y - epsilon
        Z_m, _, _, _ = generate_gaussian_distribution(row, col, mu_x, mu_y, sigma_x, sigma_y_m, rho)
    elif param_name == "mu_x":
        mu_x_m = mu_x - epsilon
        Z_m, _, _, _ = generate_gaussian_distribution(row, col, mu_x_m, mu_y, sigma_x, sigma_y, rho)
    elif param_name == "mu_y":
        mu_y_m = mu_y - epsilon
        Z_m, _, _, _ = generate_gaussian_distribution(row, col, mu_x, mu_y_m, sigma_x, sigma_y, rho)
    elif param_name == "rho":
        rho_m = rho - epsilon
        Z_m, _, _, _ = generate_gaussian_distribution(row, col, mu_x, mu_y, sigma_x, sigma_y, rho_m)
    loss_m = np.sum(convolution(test_img, Z_m)[1])
    
    # 数值梯度
    numerical_grad = (loss_p - loss_m) / (2 * epsilon)
    
    # 解析梯度
    par_sigma_1, par_sigma_2, par_mu_x, par_mu_y, par_rho = partial_derivatives(
        Q_orig, conv_rst_orig, Xc_orig, Yc_orig, mu_x, mu_y, sigma_x, sigma_y, rho
    )
    
    # 映射参数名到解析梯度
    gradient_map = {
        "sigma_x": par_sigma_1,
        "sigma_y": par_sigma_2,
        "mu_x": par_mu_x,
        "mu_y": par_mu_y,
        "rho": par_rho
    }
    analytical_grad = gradient_map[param_name]
    
    # 输出结果
    print(f"Parameter: {param_name}")
    print(f"Numerical Gradient: {numerical_grad:.6e}")
    print(f"Analytical Gradient: {analytical_grad:.6e}")
    print(f"Difference: {abs(numerical_grad - analytical_grad):.6e}")
    print("="*40)

    # 测试所有参数

def show_peaks(ori_img, peaks):
    '''
    plot peaks to show how good the performance of our method.
    '''
    row, col = ori_img.shape
    Residual = ori_img.copy() / (np.max(ori_img)+1e-5)
    Residual = Residual - np.mean(Residual)
    _, axs = plt.subplots(1, 3)
    axs[0].imshow(ori_img)
    axs[0].set_title('Original Image')
    figure1 = np.zeros(img.shape)
    for idx, items in enumerate(peaks):
        tgt_zs = generate_gaussian_distribution(row, col, *items[:-1])[0]
        figure1 += tgt_zs/np.max(tgt_zs) * items[-1]
        Residual = Residual - convolution(Residual, tgt_zs/np.max(tgt_zs))[1]
        circle = matplotlib.patches.Ellipse((items[0], items[1]), items[2], items[3], color='r', fill=False)
        axs[1].add_patch(circle)
        axs[1].text(items[0], items[1],f"{idx+1}",c='r',ha='center',va='center')
    axs[1].imshow(figure1)
    axs[1].set_title('Fitted Sets of Gaussians')
    axs[2].imshow(Residual)
    axs[2].set_title(f'Residual Image, max_resudual = {np.max(Residual)}')
    plt.show()

def generate_sample_image(shape=(100, 100)):
    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    
    # 第一个高斯峰
    gauss1 = 100 * np.exp(-((x-30)**2/(2*10**2) + (y-40)**2/(2*15**2)))
    # 第二个高斯峰
    gauss2 = 80 * np.exp(-((x-70)**2/(2*8**2) + (y-60)**2/(2*12**2)))
    
    # 添加噪声
    noise = np.random.normal(0, 2, shape)
    return gauss1 + gauss2 + noise


def generate_anisotropic_gaussian(shape=(100,100)):
    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    gauss = 100 * np.exp(-((x-50)**2/(2*5**2) + (y-50)**2/(2*20**2)))  # σx=5, σy=20
    return gauss

if __name__ == "__main__":
    img = generate_sample_image()


    # we use a randomly chosen image to test if our function could behave well.
    # test_img_path = "E:\\Li Lab\\itti_and_lif\\VAL_results\\non_opt_val\\0615\\0020.png"

    test_img_path = "E:\\Li Lab\\itti_and_lif\\video\\annotation\\0001\\maps\\0001.png"
    img = cv2.imread(test_img_path, cv2.IMREAD_GRAYSCALE)
    #peaks = loop_for_multiple_gaussian_fitting(img)
    #show_peaks(img, peaks)
    
    # 固定随机种子
    np.random.seed(42)
    test_img = generate_test_image()
    row, col = test_img.shape
    numerical_gradient("sigma_x")
    numerical_gradient("sigma_y")
    numerical_gradient("mu_x")
    numerical_gradient("mu_y")
    numerical_gradient("rho")

    #img = generate_anisotropic_gaussian()
    peaks = loop_for_multiple_gaussian_fitting(img)
    print("Fitted parameters:", peaks[0][:5])
    show_peaks(img, peaks)
    loop_for_multiple_gaussian_fitting2(img)