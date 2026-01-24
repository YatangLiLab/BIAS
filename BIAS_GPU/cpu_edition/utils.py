import cv2
import argparse
import numpy as np
# import cupy as np
import matplotlib.pyplot as plt
from scipy.ndimage import maximum_filter
from scipy.signal import convolve2d
from scipy.signal import convolve
import time
from functools import wraps
import ctypes

def load_maximum_dll(path = './find_local_maximas.dll'):
    maximum_dll = ctypes.CDLL(path)
    return maximum_dll

def find_maximum_mat(image,maximum_dll):
    assert len(image.shape) == 2
    image = np.float32(image)/(np.max(image)+1e-6)
    M,N = image.shape
    result = np.zeros((M,N),dtype=np.uint8)
    image_ptr = image.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    result_ptr = result.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
    maximum_dll.find_local_maximas_wrapper.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_uint8), ctypes.c_int, ctypes.c_int]
    maximum_dll.find_local_maximas_wrapper(image_ptr, result_ptr, M, N)
    #plt.imshow(result)
    #plt.show()
    return result

def timing_decorator(func):
    """timer"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()  
        result = func(*args, **kwargs)  
        end_time = time.time()  
        print(f"{func.__name__} running time: {end_time - start_time} second(s)")
        return result
    return wrapper
# @timing_decorator
def specific_Gabor_filter(gabor_lib,image_ptr,M,N,Orientation,Frequency,kernal_size):
    """use dll to compute, you may check the computation cost here."""
    gabor_lib.gabor_filter(image_ptr, M, N, Orientation,Frequency,kernal_size)

def initialize_gabor_lib(path = "./gabor_filter.dll"):
    """ 
    if you edited my code, and you find that the code raise an error that it could not find the dll file, it might be triggerd by C++ function or class in the cpp file like iostream/string/vector/...

    use ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.c_int as input datatype.
    
    return a gabor_lib, use it just like: gabor_lib.gabor_filter(image_ptr, int: M, int: N, float: Orientation, float: Frequency, int: kernal_size)
    """
    # if path[-4::] != ".dll":
    #     raise AssertionError("maybe you should check if you use the right pathway. the end of the path is not .dll.")
    gabor_lib = ctypes.cdll.LoadLibrary(path)
    gabor_lib.gabor_filter.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.c_int]
    return gabor_lib


def gabor_filter(gabor_lib:ctypes.CDLL,image:np.float32, Orientation:float, Frequency:float,kernal_size:int):
    """ the core function. input gabor lib, the image and other essential parameters so the code can run."""
    image_ptr = image.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    # use Cpp DLL
    M, N = image.shape
    try:
        specific_Gabor_filter(gabor_lib,image_ptr, M, N, Orientation,Frequency,kernal_size)
    except Exception as e:
        print(f"Error: {e}")
        raise RuntimeError(e)

def gabor_kernel(frequency, ksize,theta):
    """Generate a 1D Gabor kernel."""
    sigma = 2 * np.pi**2 / frequency
   
    # Generate the kernel
    t = np.linspace(-ksize, ksize, 2 * ksize + 1)
    S_sigma = np.exp(-(t - ksize)**2 / (2 * sigma**2))
    cos_c = np.cos((t - ksize) * theta)
    sin_c = np.sin((t - ksize) * theta)
    
    return np.array([S_sigma * (cos_c + 1j * sin_c)])

def gabor_conv1d(image:np.float32, orientation:float, frequency:float, ksize:int):
    """Apply a 1D Gabor filter to an image."""
    M, N = image.shape
    sigma = 2 * np.pi**2 / frequency
    w_c_theta = frequency * np.cos(orientation)
    w_s_theta = frequency * np.sin(orientation)
    
    # Create the Gabor kernel
    kernel_x = gabor_kernel(frequency, ksize,w_c_theta)
    kernel_y = gabor_kernel(frequency, ksize,w_s_theta).T
    
    # Apply the kernel horizontally
    #RJ = np.zeros((M, N), dtype=np.complex128)
    #IJ = np.zeros((M, N), dtype=np.complex128)
    #for y in range(M):
    #    RJ[y, :] = convolve2d(image[y, :], kernel_x, mode='same', boundary='wrap')
    #    IJ[y, :] = convolve2d(image[y, :] * 1j, kernel_x, mode='same', boundary='wrap')
    #RJ = convolve2d(image,kernel_x,mode='same', boundary='wrap')
    #IJ = convolve2d(image * 1j, kernel_x, mode='same', boundary='wrap')
    RJ = convolve(image,kernel_x,mode='same')
    IJ = convolve(image * 1j, kernel_x, mode='same')
    # Apply the kernel vertically
    #F = np.zeros((M, N), dtype=np.float32)
    #IF = np.zeros((M, N), dtype=np.float32)
    #for x in range(N):
    #    F[:, x] = convolve2d(RJ[:, x] + 1j * IJ[:, x], kernel_y, mode='same', boundary='wrap')
    #    IF[:, x] = convolve2d(RJ[:, x] - 1j * IJ[:, x], kernel_y, mode='same', boundary='wrap')
    #F = convolve2d(RJ + 1j * IJ, kernel_y, mode='same', boundary='wrap')
    #IF = convolve2d(RJ - 1j * IJ, kernel_y, mode='same', boundary='wrap')
    F = convolve2d(RJ + 1j * IJ, kernel_y, mode='same')
    IF = convolve2d(RJ - 1j * IJ, kernel_y, mode='same')
    # Compute the magnitude of the complex result
    conv_img = np.sqrt(F**2 + IF**2)
    
    return conv_img

def gabor_conv2d(image:np.float32, orientation:float, frequency:float, ksize:int):
    kernel = cv2.getGaborKernel((ksize,ksize),sigma=3,theta=orientation,lambd=4,gamma=1,psi=0,ktype=cv2.CV_32F)
    return convolve2d(image,kernel, mode='same')

def fill_in_it(fillIn_lib,saliency_ptr,image_ptr,M,N,image_mean,image_sd,threshold=0.03,max_iteration=50):
    """use dll to compute, you may check the computation cost here."""
    fillIn_lib.update_saliency_map(saliency_ptr,image_ptr,M,N,image_mean,image_sd,threshold,max_iteration)

def initialize_fillIn_lib(path = "./fillin.dll"):
    """ 
    if you edited my code, and you find that the code raise an error that it could not find the dll file, it might be triggerd by C++ function or class in the cpp file like iostream/string/vector/...

    use ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.c_int as input datatype.
    
    return a initialize_fillIn_lib, use it just like: (fillIn_lib, saliency_img, original_img)
    """
    if path[-4::] != ".dll":
        raise AssertionError("maybe you should check if you use the right pathway. the end of the path is not .dll.")
    fillIn_lib = ctypes.cdll.LoadLibrary(path)
    fillIn_lib.update_saliency_map.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int, ctypes.c_float,ctypes.c_float,ctypes.c_float, ctypes.c_int]
    return fillIn_lib


def fill_blank(fillIn_lib, saliency_img, original_img):
    """ the core function. input gabor lib, the image and other essential parameters so the code can run."""
    #print(saliency_img.shape)
    #print(original_img.shape)
    assert saliency_img.shape == original_img.shape
    saliency_img = np.float32(saliency_img)/255#np.float32(saliency_img)/255
    original_img = np.float32(original_img)/255#np.float32(original_img)/255
    saliency_ptr = saliency_img.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    image_ptr = original_img.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    # use Cpp DLL
    M, N = saliency_img.shape
    try:
        fill_in_it(fillIn_lib,saliency_ptr,image_ptr,M,N,np.mean(original_img),np.std(original_img,ddof=0))
    except Exception as e:
        print(f"Error: {e}")
        raise RuntimeError(e)
    return saliency_img
    
def gaussian_pyrimid(image):
    """
    return /2 resolution gaussian pyrimid
    """
    #image = cv2.GaussianBlur(image,(5,5),0.1 * image.shape[1],sigmaY=0.1*image.shape[0])
    #return cv2.resize(image,(image.shape[1]//2+1,image.shape[0]//2+1),interpolation=cv2.INTER_NEAREST) 
    return cv2.pyrDown(image)

def eight_pyrimid_built(image):
    """
    yielding horizontal and vertical image-reduction factors ranging from 1:1 (scale zero) to 1:256 (scale eight) in eight octaves.
    """
    image_list = []
    image2 = image.copy()
    for i in range(8):
        image_list.append(image2)
        image2 = gaussian_pyrimid(image2)
    # fig, axs = plt.subplots(4,2)
    # for i in range(8):
    #     axs[i//2,i%2].imshow(image_list[i])
    # plt.show()
    return image_list

def subtraction(img1,img2,ifshow = False):
    """
    return img1 - img2 but resized to the higher resolution
    """
    if np.max(img1) == 0 and np.max(img2) == 0:
        return img1 if img1.shape[0] > img2.shape[0] else img2
    _shape = img1.shape if img1.shape[0] >= img2.shape[0] else img2.shape
    shape = (_shape[1],_shape[0])
    image_1 = img1 if img1.shape == _shape else cv2.resize(img1,shape,interpolation=cv2.INTER_LINEAR)
    image_2 = img2 if img2.shape == _shape else cv2.resize(img2,shape,interpolation=cv2.INTER_LINEAR)
    image_1 = np.float32(image_1)
    image_2 = np.float32(image_2)
    if ifshow:
        fig, axs = plt.subplots(2,3)
        axs[0,0].imshow(image_1)
        axs[0,0].set_title("image 1")
        axs[0,1].imshow(image_2)
        axs[0,1].set_title("image 2")
        axs[1,0].imshow(image_1 - image_2)
        axs[1,0].set_title("image 1 - image 2")
        axs[1,1].imshow(img2)
        axs[1,1].set_title("original img2")
        axs[1,2].imshow(img1)
        axs[1,2].set_title("original img1")
        plt.show()
    return image_1 - image_2 

def sub_img2(img1,img2,ifshow = False):
    """
    return img1 - img2 but resized to the higher resolution
    """
    if ifshow:
        plt.imshow(img1)
        plt.show()
        plt.imshow(img2)
        plt.show()
    _shape = img1.shape if img1.shape[0] >= img2.shape[0] else img2.shape
    shape = (_shape[1],_shape[0])
    image_1 = img1 if img1.shape == _shape else cv2.resize(img1,shape,interpolation=cv2.INTER_LINEAR)
    image_2 = img2 if img2.shape == _shape else cv2.resize(img2,shape,interpolation=cv2.INTER_LINEAR)
    rst = np.where(image_2 > 0,image_1 - image_2/(np.max(image_2)+1), image_1 ) 
    return np.where(rst>0,rst,0)


def addition(img1,img2,shape = "Not_given"):
    """
    through calculate we have fourth shape
    """
    if shape == "Not_given":
        _shape = img1.shape if img1.shape[0] >= img2.shape[0] else img2.shape
        shape = (_shape[1],_shape[0])
    image1 = img1 if img1.shape == _shape else cv2.resize(img1,shape,interpolation=cv2.INTER_LINEAR)
    image2 = img2 if img2.shape == _shape else cv2.resize(img2,shape,interpolation=cv2.INTER_LINEAR)
    image1 = np.float32(image1)
    image2 = np.float32(image2)
    return image1 + image2

def multation(img1,img2,shape = "Not_given"):
    """
    through calculate we have fourth shape
    """
    if shape == "Not_given":
        _shape = img1.shape if img1.shape[0] >= img2.shape[0] else img2.shape
        shape = (_shape[1],_shape[0])
    image1 = img1 if img1.shape == _shape else cv2.resize(img1,shape,interpolation=cv2.INTER_LINEAR)
    image2 = img2 if img2.shape == _shape else cv2.resize(img2,shape,interpolation=cv2.INTER_LINEAR)
    image1 = np.float32(image1)
    image2 = np.float32(image2)
    return image1 * image2

def not_normalize_img(img,maximum_dll,M=1):
    """
    normalize img scale to 0~M, globally multiply it by $(M-\\bar{m})^2$
    """
    # print("This")
    image = (img - np.min(img)) / (np.max(img) - np.min(img)) * M if (np.max(img) - np.min(img)) > 0 else img*0
    w,h = image.shape
    maxima = maximum_filter(image, size=(max(w/10,3),max(h/10,3)))
    maxima = (image == maxima) # a 0-1 sparce mat.
    mnum = maxima.sum()
    maxima = np.multiply(maxima, image)
    mbar = float(maxima.sum()) / (mnum + 1e-3) if mnum else 0
    #print(mbar)
    #fig, axs = plt.subplots(1,2)
    #axs[0].imshow(img)
    #axs[1].imshow(maxima)
    #plt.show()
    return image*((M - mbar)**2)
    # local_max = maximum_filter(image, size=3) == image
    # # Add a threshold
    # local_max &= image > 0.2 # as threshold
    # local_max_num = np.count_nonzero(local_max)
    # local_max_sum = np.sum(image[local_max])
    # local_max_avg = local_max_sum / local_max_num if local_max_num > 0 else 0
    # if local_max_num > 1:
    #     p = (1 - local_max_avg) ** 2
    # else:
    #     p = 1
    # return image * p

def normalize_img(img,maximum_dll,M=1):
    """
    normalize img scale to 0~M, globally multiply it by (M-\\bar{m})^2
    """
    # img = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-5) * M if np.max(img) - np.min(img) >= 1e-5 else img*0
    
    if np.max(img) == 0:
        if np.min(img) == 0:
            return img
        else:
            raise RuntimeError("strange things happens here")
    # img = np.abs(img - np.mean(img))
    img = (img - np.min(img)) / (np.max(img) - np.min(img)) * M if np.max(img) - np.min(img) else img*0
    #w,h = image.shape
    local_maximum_mat = find_maximum_mat(img,maximum_dll)
    mnum = local_maximum_mat.sum()
    if mnum == 0:
        return local_maximum_mat
    # fig, axs = plt.subplots(1,2)
    # axs[0].imshow(img)
    # axs[1].imshow(local_maximum_mat)
    # plt.show()
    maxima = img * local_maximum_mat

    mbar = float(maxima.sum()) / (mnum+1e-5) if mnum else 0
    return img*(M - mbar)**2

def conv_function(image:np.uint8,kernel:np.matrix)->np.float32:
    """
    generate a totally same size convolution image. Boundary use symm boundary.
    """
    # return cv2.filter2D(image,-1,kernel,borderType=cv2.BORDER_WRAP)
    return convolve2d(image, kernel, mode='same', boundary='symm')

def read_video(path):
    """
    just as the name said, read a video
    """
    cap = cv2.VideoCapture(path)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    final_shape = (width,height)
    return cap, final_shape

def pre_processing(image:np.uint8, args:argparse.Namespace)->np.float16:
    """
    do some essential pre-processing jobs like generate gray image, change shape
    """
    return cv2.cvtColor(cv2.resize(image,args.default_size),cv2.COLOR_BGR2GRAY)


def get_gaussian_kernal(ksize:int, sigma:float , dim:int = 2) -> np.ndarray:
    """
    create a gaussian kernal (could be 1 dimension or 2)
    """
    gaussian_1dkernal = cv2.getGaussianKernel(ksize,sigma)
    if dim == 1:
        return gaussian_1dkernal
    elif dim == 2:
        return gaussian_1dkernal * gaussian_1dkernal.T
    else:
        raise NotImplementedError

def interference_function(current_Saliency_map, former_fixation_point,mode = "Gaussian", boxwidth_param = 0.2):
    """
    define a function change the saliency map weight, using different adjust mode including gaussian, box or else
    """
    mat_shape = current_Saliency_map.shape
    #print(mat_shape)
    #print(former_fixation_point)
    # print(mat_shape)
    if mode == "Gaussian":
        long_gaussian = get_gaussian_kernal(max(mat_shape)*2,max(mat_shape)/2,dim=1)
        short_gaussian = get_gaussian_kernal(min(mat_shape)*2,min(mat_shape)/2,dim=1)
        #Gaussian_kernal = get_gaussian_kernal(max(mat_shape)*2,min(mat_shape)/2)
        Gaussian_kernal =  short_gaussian * long_gaussian.T
        # plt.imshow(Gaussian_kernal)
        # plt.show()
        adjust_mat = Gaussian_kernal[min(mat_shape)-former_fixation_point[0]:min(mat_shape)+mat_shape[0]-former_fixation_point[0], \
                                     max(mat_shape)-former_fixation_point[1]:max(mat_shape)+mat_shape[1]-former_fixation_point[1]]
        adjust_mat /= np.max(adjust_mat)
        weight_mat = adjust_mat * current_Saliency_map
        # fig,((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2)
        # ax1.imshow(current_Saliency_map)
        # ax2.imshow(Gaussian_kernal)
        # ax3.imshow(adjust_mat)
        # ax4.imshow(weight_mat)
        # plt.show()
        ##print(adjust_mat)
        #print(adjust_mat.shape)
        
        #plt.imshow(adjust_mat)
        #plt.title("adjust Map")
        #plt.show()
    elif mode == "box":
        adjust_mat = np.ones_like(current_Saliency_map) * 0.1
        upper_bound = int(max(0,former_fixation_point[0] - boxwidth_param * mat_shape[0]))
        lower_bound = int(min(mat_shape[0], former_fixation_point[0] + boxwidth_param * mat_shape[0]))
        left_bound  = int(max(0,former_fixation_point[1] - boxwidth_param * mat_shape[1]))
        right_bound = int(min(mat_shape[1], former_fixation_point[1] + boxwidth_param * mat_shape[1]))
        adjust_mat[upper_bound:lower_bound,left_bound:right_bound] += 0.9
        #plt.imshow(adjust_mat)
        #plt.title("adjust Map")
        #plt.show()
        #plt.imshow(adjust_mat)
        #plt.show()
        weight_mat = adjust_mat * current_Saliency_map
    elif mode =="None":
        weight_mat = current_Saliency_map
    else:
        raise NotImplementedError
    return weight_mat
