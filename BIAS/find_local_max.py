import numpy as np
import ctypes
import cv2
import matplotlib.pyplot as plt

def load_maximum_dll(path = '.\\find_local_maximas.dll'):
    maximum_dll = ctypes.CDLL(path)
    return maximum_dll

def find_maximum_mat(image,maximum_dll):
    image = np.uint8(image/(1e-6+np.max(image))*255)
    M,N = image.shape
    result = np.zeros((M,N),dtype=np.uint8)
    image_ptr = image.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
    result_ptr = result.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
    maximum_dll.find_local_maximas_wrapper.argtypes = [ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(ctypes.c_uint8), ctypes.c_int, ctypes.c_int]
    maximum_dll.find_local_maximas_wrapper(image_ptr, result_ptr, M, N)
    return result

# 加载 DLL
if __name__ == "__main__":
    dll = ctypes.CDLL('.\\find_local_maximas.dll')

# 定义输入输出的数据类型
#M, N = 4, 4  # 矩阵的维度
#image = np.array([[1, 2, 3, 4], [4, 9, 6, 7],[4, 5, 6,7], [3, 2, 1,0]], dtype=np.float32)
#result = np.zeros((M, N), dtype=np.uint8)
    #image = cv2.imread("test_image\\im_14.jpg")
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    #image = cv2.resize(image, (image.shape[1]//1, image.shape[0]//1))
    #image /= 255
    image = np.array([np.random.binomial(5,0.5,30)]) * np.array([np.random.binomial(5,0.5,40)]).T
    image = np.float32(image)
    M,N = image.shape
    result = np.zeros((M, N), dtype=np.uint8)
    # 获取指针
    image_ptr = image.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    result_ptr = result.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))

    # 设置函数参数类型
    dll.find_local_maximas_wrapper.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_uint8), ctypes.c_int, ctypes.c_int]

    # 调用 C 函数
    dll.find_local_maximas_wrapper(image_ptr, result_ptr, M, N)
    fig, (ax1,ax2) = plt.subplots(1,2)
    ax1.imshow(image)
    ax2.imshow(result)
    plt.show()
    # 打印结果
    #print(result)