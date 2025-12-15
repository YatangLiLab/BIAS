import numpy as np
import cv2
import matplotlib.pyplot as plt
import argparse
from utils import *

def O_c_s_theta(Os,c,s,theta):
       tmp = np.abs(subtraction(Os[c][theta//45], Os[s][theta//45]))
       return tmp

def Orientation_maps(Os,args,ifshow=True):
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

def synthesis_Orientation_map(O_dict,addition_shape,args,ifshow=True):
    c_set = args.center
    delta_set = args.surrounding
    theta_set = (0,45,90,135)
    O_bar_0 = np.zeros((1,1))
    O_bar_45 = np.zeros((1,1))
    O_bar_90 = np.zeros((1,1))
    O_bar_135 = np.zeros((1,1))
    for c in c_set:
        for delta in delta_set:
            O_bar_0 = addition(O_bar_0,normalize_img(O_dict[(c,c+delta,0)]),addition_shape)
            O_bar_45 = addition(O_bar_45,normalize_img(O_dict[(c,c+delta,45)]),addition_shape)
            O_bar_90 = addition(O_bar_90,normalize_img(O_dict[(c,c+delta,90)]),addition_shape)
            O_bar_135 = addition(O_bar_135,normalize_img(O_dict[(c,c+delta,135)]),addition_shape)
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
        O_bar = addition(O_bar,normalize_img(O_bar_theta),addition_shape)
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


def read_image(path:str,default_size:tuple)->np.uint8:
    """
    just simply read an image and convert it into default_size
    """
    return cv2.cvtColor(cv2.resize(cv2.imread(path),default_size,interpolation=cv2.INTER_NEAREST),cv2.COLOR_BGR2GRAY)


def parse_args():
    parse = argparse.ArgumentParser(description='Essential parameters for gabor processing') 
    parse.add_argument('--image_path', default="E:\\your_path\\report_slides\\oldreport\\test_images\\standard.jpg", type=str, help='path of sample image') 
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

if __name__ == "__main__":
    args = parse_args()
    sample_image = read_image(args.image_path,args.default_size)
    plt.imshow(sample_image)
    plt.show()
    print(sample_image.shape)
    pyra_imgs = lst_gaussian_pyramid(sample_image,args.pyramid_height)
    fig, axs = plt.subplots(2,3)
    for index in range(args.pyramid_height):
        axs[index//3,index%3].imshow(pyra_imgs[index])
    plt.show()
    gabor_kernel_size, mini_sigma, num_of_thetas, gabor_lambda, gabor_gamma = args.gabor_kernel_size,args.mini_sigma,args.num_of_thetas,args.gabor_lambda,args.gabor_gamma
    Os = []
    for i in range(args.total_height):
        if i < args.pyramid_height:
            fig, axs = plt.subplots(2,2)
            gabor_kernels = get_gabor_kernels(gabor_kernel_size, mini_sigma, num_of_thetas, gabor_lambda, gabor_gamma)
        else:
            print(mini_sigma)
            print(gabor_lambda)
            mini_sigma *= 2
            gabor_lambda *= 2
            # fig, axs = plt.subplots(2,2)
            gabor_kernels = get_gabor_kernels(gabor_kernel_size, mini_sigma, num_of_thetas, gabor_lambda, gabor_gamma)
        #for num, id in enumerate([(0,0),(0,1),(1,0),(1,1)]):
        #        axs[id[0],id[1]].imshow(gabor_kernels[num])
        #plt.show()
        #plt.close()
        #fig, axs = plt.subplots(2,2)
        new_lst = []
        for idx1 in range(2):
            for idx2 in range(2):
                new_img = np.abs(np.float32(conv_function(pyra_imgs[min(i,args.pyramid_height-1)],gabor_kernels[2*idx1 + idx2])))
                new_lst.append(new_img / (np.mean(new_img)+1e-3))
        #         axs[idx1,idx2].imshow(new_lst[-1])
        # plt.show()
        # plt.close()
        Os.append(new_lst.copy())
    #for i in range(len(Os)):
    #    plt.imshow(Os[i][0])
    #    plt.show()
    O_dict = Orientation_maps(Os,args)
    Obar = synthesis_Orientation_map(O_dict,Os[0][0].T.shape,args,False)
    plt.imshow(Obar)
    plt.show()
    