import numpy as np
import cv2
import matplotlib.pyplot as plt
import argparse
from utils import normalize_img, subtraction, addition, initialize_gabor_lib, gabor_filter, timing_decorator
from utils import gabor_conv1d, timing_decorator, load_maximum_dll, gabor_conv2d
import copy
import ctypes

def O_c_s_theta(Os,c,s,theta):
       tmp = np.abs(subtraction(Os[theta][c], Os[theta][s]))
    #    plt.imshow(Os[c][theta//45])
    #    plt.show()
    #    plt.imshow(Os[s][theta//45])
    #    plt.show()
       #tmp /= np.max(np.abs(tmp))
       #tmp *= 254
       #print(c)
       #print(s)
       #print(theta)
       #plt.imshow(tmp)
       #plt.show()
       return tmp

def Orientation_maps(Os,args,ifshow=False):
    c_set = args.center
    delta_set = args.surrounding
    theta_set = (0,45,90,135)
    lists = []
    for c in c_set:
        for d in delta_set:
            lists.append((c,c+d))
    O_dict = {}
    for c in c_set:
        for delta in delta_set:
            for theta in theta_set:
                O_dict[(c,c+delta,theta)] = O_c_s_theta(Os,c,c+delta,theta)
    if ifshow:
        fig, axs = plt.subplots(len(theta_set),len(lists))
        for it, t in enumerate(theta_set):
            for il,l in enumerate(lists):
                if len(lists) != 1:
                    axs[it,il].imshow(O_dict[l[0],l[1],t])
                else:
                    axs[it].imshow(O_dict[l[0],l[1],t])
        plt.show()
    return O_dict

def synthesis_Orientation_map(O_dict,args,norm_dll,ifshow=False):
    c_set = args.center
    delta_set = args.surrounding
    theta_set = (0,45,90,135)
    O_bar_0 = np.zeros((1,1))
    O_bar_45 = np.zeros((1,1))
    O_bar_90 = np.zeros((1,1))
    O_bar_135 = np.zeros((1,1))
    for c in c_set:
        for delta in delta_set:
            O_bar_0 = addition(O_bar_0,normalize_img(O_dict[(c,c+delta,0)],norm_dll))
            O_bar_45 = addition(O_bar_45,normalize_img(O_dict[(c,c+delta,45)],norm_dll))
            O_bar_90 = addition(O_bar_90,normalize_img(O_dict[(c,c+delta,90)],norm_dll))
            O_bar_135 = addition(O_bar_135,normalize_img(O_dict[(c,c+delta,135)],norm_dll))
            if ifshow:
                fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2)
                ax1.imshow(O_dict[(c,c+delta,0)])
                ax1.set_title("verticle(0 degrees)")
                ax2.imshow(O_dict[(c,c+delta,45)])
                ax2.set_title("upright(45 degrees)")
                ax3.imshow(O_dict[(c,c+delta,90)])
                ax3.set_title("horizontal(90 degrees)")
                ax4.imshow(O_dict[(c,c+delta,135)])
                ax4.set_title("downright(135 degrees)")
                plt.show()

    O_bar = np.zeros((1,1))
    for O_bar_theta in [O_bar_0, O_bar_45, O_bar_90, O_bar_135]:
        O_bar = addition(O_bar,normalize_img(O_bar_theta,norm_dll))
    return O_bar

def lst_gaussian_pyramid(image:np.uint8,length:int) -> list[np.uint8]:
    """
    get np.uint8 image as input, generate a list containing a list containing gaussian pyramids of the image
    """
    img = image.copy()
    img_lst = []
    for _height in range(length):
        img_lst.append(img)
        img = cv2.pyrDown(img)
    return img_lst


def implement_gaborfilter(gabor_lib:ctypes.CDLL,image:np.float32, Orientation:float, Frequency:float,kernal_size:int):
    gabor_filter(gabor_lib,image, Orientation, Frequency,kernal_size)
    # gabor_conv1d(image,Orientation,Frequency,kernal_size)
    # gabor_conv2d(image,Orientation,Frequency,kernal_size)


def read_image(path:str,default_size:tuple)->np.uint8:
    """
    just simply read an image and convert it into default_size
    """
    return cv2.cvtColor(cv2.resize(cv2.imread(path),default_size,interpolation=cv2.INTER_NEAREST),cv2.COLOR_BGR2GRAY)
@timing_decorator
def test_Orientation_pyramid_build(image_pyramid:list[np.uint8], total_height, pyramid_height,gabor_lib,args, ifshow = False):
    """
    lets just guess the source image is np.uint8 image, after some processing problems, we got the intensity pyramid(or we could send color channel into it someday, but not now.)
    the first thing we should notice is that the usually image should be interpolated as default size, which is 640 * 480.but actually we just use c = 2/3/4, there are actually no needs to filter it in such a detail.
    what we have to do is:
    
    first, do some deep-copy incase change all the results.
    
    second, use the 2/3/4/... items to compute gabor results.

    finally, return gabor results.
    """
    # image_pyramid[0] = 0
    # cost less than 1e10-5 seconds = 0.01 ms
    calculating_parts = []
    for c in args.center:
        for d in args.surrounding:
            calculating_parts.append(c)
            calculating_parts.append(c+d)
    calculating_parts = set(calculating_parts)
    # first we just want to get away from copy the most computation-cost image
    for index in range(1,len(image_pyramid)):
        image_pyramid[index] = np.float32(image_pyramid[index])/255
    vert_degree_lst = copy.deepcopy(image_pyramid)
    hori_degree_lst = copy.deepcopy(image_pyramid)
    upright_lst = copy.deepcopy(image_pyramid)
    downright_lst = copy.deepcopy(image_pyramid)

    default_freq = 2 * np.pi ** 2 / 2.7
    kernal_size = 4

    for index in range(1,pyramid_height):
        if not (index in calculating_parts):
            continue
        implement_gaborfilter(gabor_lib,vert_degree_lst[index],0.0,default_freq,kernal_size)
        implement_gaborfilter(gabor_lib,hori_degree_lst[index],np.pi/2,default_freq,kernal_size)
        implement_gaborfilter(gabor_lib,upright_lst[index],np.pi/4,default_freq,kernal_size)
        implement_gaborfilter(gabor_lib,downright_lst[index],np.pi*3/4,default_freq,kernal_size)
    for index in range(pyramid_height,total_height):
        if not (index in calculating_parts):
            continue
        default_freq /= 2
        kernal_size = round(np.pi / default_freq) + 1
        vert_degree_lst[index] = copy.copy(image_pyramid[pyramid_height-1])
        hori_degree_lst[index] = copy.copy(image_pyramid[pyramid_height-1])
        upright_lst[index] = copy.copy(image_pyramid[pyramid_height-1])
        downright_lst[index] = copy.copy(image_pyramid[pyramid_height-1])

        implement_gaborfilter(gabor_lib,vert_degree_lst[index],0.0,default_freq,kernal_size)
        implement_gaborfilter(gabor_lib,hori_degree_lst[index],np.pi/2,default_freq,kernal_size)
        implement_gaborfilter(gabor_lib,upright_lst[index],np.pi/4,default_freq,kernal_size)
        implement_gaborfilter(gabor_lib,downright_lst[index],np.pi*3/4,default_freq,kernal_size)
    Os = {0:vert_degree_lst,90:hori_degree_lst,45:upright_lst,135:downright_lst}
    if ifshow:
        fig, axs = plt.subplots(4,total_height - 1)
        for row in range(4):
            for col in range(1,total_height):
                axs[row,col-1].imshow(Os[row * 45][col])
        plt.show()
    return Os

def Orientation_pyramid_build(image_pyramid:list[np.uint8], total_height, pyramid_height,gabor_lib,args, ifshow = False):
    """
    lets just guess the source image is np.uint8 image, after some processing problems, we got the intensity pyramid(or we could send color channel into it someday, but not now.)
    the first thing we should notice is that the usually image should be interpolated as default size, which is 640 * 480.but actually we just use c = 2/3/4, there are actually no needs to filter it in such a detail.
    what we have to do is:
    
    first, do some deep-copy incase change all the results.
    
    second, use the 2/3/4/... items to compute gabor results.

    finally, return gabor results.
    """
    # image_pyramid[0] = 0
    # cost less than 1e10-5 seconds = 0.01 ms
    calculating_parts = []
    for c in args.center:
        for d in args.surrounding:
            calculating_parts.append(c)
            calculating_parts.append(c+d)
    calculating_parts = set(calculating_parts)
    # first we just want to get away from copy the most computation-cost image
    for index in range(1,len(image_pyramid)):
        image_pyramid[index] = np.float32(image_pyramid[index])/255
    vert_degree_lst = copy.deepcopy(image_pyramid)
    hori_degree_lst = copy.deepcopy(image_pyramid)
    upright_lst = copy.deepcopy(image_pyramid)
    downright_lst = copy.deepcopy(image_pyramid)

    default_freq = 2 * np.pi ** 2 / 2.7
    kernal_size = 4

    for index in range(1,pyramid_height):
        if not (index in calculating_parts):
            continue
        gabor_filter(gabor_lib,vert_degree_lst[index],0.0,default_freq,kernal_size)
        gabor_filter(gabor_lib,hori_degree_lst[index],np.pi/2,default_freq,kernal_size)
        gabor_filter(gabor_lib,upright_lst[index],np.pi/4,default_freq,kernal_size)
        gabor_filter(gabor_lib,downright_lst[index],np.pi*3/4,default_freq,kernal_size)
    for index in range(pyramid_height,total_height):
        if not (index in calculating_parts):
            continue
        default_freq /= 2
        kernal_size = round(np.pi / default_freq) + 1
        vert_degree_lst[index] = copy.copy(image_pyramid[pyramid_height-1])
        hori_degree_lst[index] = copy.copy(image_pyramid[pyramid_height-1])
        upright_lst[index] = copy.copy(image_pyramid[pyramid_height-1])
        downright_lst[index] = copy.copy(image_pyramid[pyramid_height-1])

        gabor_filter(gabor_lib,vert_degree_lst[index],0.0,default_freq,kernal_size)
        gabor_filter(gabor_lib,hori_degree_lst[index],np.pi/2,default_freq,kernal_size)
        gabor_filter(gabor_lib,upright_lst[index],np.pi/4,default_freq,kernal_size)
        gabor_filter(gabor_lib,downright_lst[index],np.pi*3/4,default_freq,kernal_size)
    Os = {0:vert_degree_lst,90:hori_degree_lst,45:upright_lst,135:downright_lst}
    if ifshow:
        fig, axs = plt.subplots(4,total_height - 1)
        for row in range(4):
            for col in range(1,total_height):
                axs[row,col-1].imshow(Os[row * 45][col])
        plt.show()
    return Os




def parse_args():
    parse = argparse.ArgumentParser(description='Essential parameters for gabor processing') 
    parse.add_argument('--image_path', default="E:\\your_path\\RealtimeSaliency\\test_image\\circle.jpg", type=str, help='path of sample image') 
    parse.add_argument('--total_height',default=9,type=int,help='total height of orientation pyramid, equals to height of gaussian pyramid + kernel size Pyramid')
    parse.add_argument('--pyramid_height', default=5, type=int, help='height of Gaussian Pyramid')
    parse.add_argument('--gabor_kernel_size',default=33,type=int,help='the minimal value of gabor kernel size. when meet some constrains, we would double some params.')
    parse.add_argument('--num_of_thetas',default=4,type=int,help='different gabor filter orientations.')
    parse.add_argument('--mini_sigma',default=0.5,type=float,help='sigma of gabor kernel, if the image is too small hori2then double it.')
    parse.add_argument('--gabor_lambda',default=np.pi/np.sqrt(2*np.log(1/0.5)),type=float,help = 'lambda for gabor kernels')
    parse.add_argument('--gabor_gamma',default=1,type=float,help = 'gamma value for gabor filter.')
    parse.add_argument('--default_size',default=(640,480),type=tuple,help = 'default size of image')
    parse.add_argument('--center',default = (2,3,4),type=tuple,help = "center params. Itti default params are (2, 3, 4)")
    parse.add_argument('--surrounding',default = (3,4),type=tuple,help = "surrounding params. Itti default params are (3, 4)")
    # parse.add_argument('--num_of_motions',default=8,type=int,help='different gabor motion filter orientations.')
    
    args = parse.parse_args() 
    return args

# @timing_decorator
def O_bar_synthesis(image_pyramid,args,gabor_lib,norm_dll):
    if np.any(image_pyramid[0]>0):
        Os = Orientation_pyramid_build(image_pyramid,args.total_height,args.pyramid_height, gabor_lib,args)
        O_dict = Orientation_maps(Os,args)
        Obar = synthesis_Orientation_map(O_dict,args,norm_dll,False)
        return Obar
    else:
        long_edge_gaussian_kernal = cv2.getGaussianKernel(image_pyramid[1].shape[0],image_pyramid[1].shape[1]/3)
        short_edge_gaussian_kernal = cv2.getGaussianKernel(image_pyramid[1].shape[1],image_pyramid[1].shape[1]/3)
        center_img = long_edge_gaussian_kernal * short_edge_gaussian_kernal.T
        center_img /= np.max(center_img)
        return np.float32(center_img)
    
def test_O_bar_synthesis(image_pyramid,args,gabor_lib,norm_dll):
    if np.any(image_pyramid[0]>0):
        Os = test_Orientation_pyramid_build(image_pyramid,args.total_height,args.pyramid_height, gabor_lib,args)
        O_dict = Orientation_maps(Os,args)
        Obar = synthesis_Orientation_map(O_dict,args,norm_dll,False)
        return Obar
    else:
        long_edge_gaussian_kernal = cv2.getGaussianKernel(image_pyramid[1].shape[0],image_pyramid[1].shape[1]/3)
        short_edge_gaussian_kernal = cv2.getGaussianKernel(image_pyramid[1].shape[1],image_pyramid[1].shape[1]/3)
        center_img = long_edge_gaussian_kernal * short_edge_gaussian_kernal.T
        center_img /= np.max(center_img)
        return np.float32(center_img)

if __name__ == "__main__":
    args = parse_args()
    sample_image = read_image(args.image_path,args.default_size)
    # plt.imshow(sample_image)
    # plt.show()
    print(sample_image.shape)
    pyra_imgs = lst_gaussian_pyramid(sample_image,args.total_height)
    # fig, axs = plt.subplots(2,3)
    # for index in range(args.pyramid_height):
    #     axs[index//3,index%3].imshow(pyra_imgs[index])
    # plt.show()
    gabor_lib = initialize_gabor_lib()
    norm_lib = load_maximum_dll()
    Obar = test_O_bar_synthesis(pyra_imgs,args,gabor_lib,norm_lib)
    plt.imshow(Obar)
    plt.show()
    