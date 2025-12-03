import cv2
import matplotlib.pyplot as plt
import numpy as np
# import cupy as np
from orientation_saliency import O_bar_synthesis
import os

from utils import *

##-----------------------##
"""
If you want to use this code, you can just \n
from Itti_method import Itti_Saliency_map\n
then you can just use it as :\n
itti_saliency_map = Itti_Saliency_map("your_image_path", ifshow = False)\n
the parameter ifshow controls that if the 6 + 12 + 24 different maps are displayed.\n
Hope you enjoy it!\n

Caution: Sometimes CV2 may not be able to read your image, please change the image format or check if your image path is correct.
"""
##-----------------------##

def resize_to_normal_shape(image):
    """
    Input is provided in the form of static color images, usually digitized at 640*480 resolution
    """
    resized_image = cv2.resize(image,(320,240),interpolation=cv2.INTER_LINEAR)
    # resized_image = cv2.resize(image,(640,480),interpolation=cv2.INTER_NEAREST)
    # resized_image = cv2.resize(image,(320,240),interpolation=cv2.INTER_NEAREST)
    return resized_image

def seperate_RGB_chanells(image):
   """
    Seperate RGB chanells to going through following algorithms
    """
   return image[:,:,0], image[:,:,1], image[:,:,2]


def Is_scale(img_lst,Darklist,c,s):
    tmp = np.float64(subtraction(img_lst[c] - Darklist[c], Darklist[s] - img_lst[s]))
    tmp -= np.mean(tmp)
    # tmp = np.add(tmp,-np.mean(tmp),casting="unsafe")
    #tmp /= np.max(np.abs(tmp))
    #tmp *= 254
    #plt.imshow(tmp)
    #plt.title(f"Is scale{c},{s}")
    #plt.show()
    return (np.float32(np.where(tmp>0,tmp,0)),np.float32(np.where(tmp<0,-tmp,0)))
    #tmp = subtraction(img_lst[c],img_lst[s])
    #return (np.int16(np.where(tmp>0,tmp,0)),np.int16(np.where(tmp<0,-tmp,0)))

def RG_scale(Rs,Gs,c,s):
    
    tmp = np.float64(subtraction(Rs[c]-Gs[c],Gs[s]-Rs[s]))
    tmp -= np.mean(tmp)
    
    #tmp /= np.max(np.abs(tmp))
    #tmp *= 254
    #return (np.uint8(np.where(tmp>0,tmp,0)),np.uint8(np.where(tmp<0,-tmp,0)))
    # fig,axs = plt.subplots(3,2)
    # axs[0,0].imshow(Rs[c])
    # axs[0,1].imshow(Gs[c])
    # axs[1,0].imshow(Rs[s])
    # axs[1,1].imshow(Gs[s])
    # axs[2,0].imshow(np.where(tmp>0,tmp,0))
    # axs[2,1].imshow(np.where(tmp<0,-tmp,0))
    # plt.show()
    return (np.float32(np.where(tmp>0,tmp,0)),np.float32(np.where(tmp<0,-tmp,0)))

def BY_scale(Bs,Ys,c,s):
    tmp = np.float64(subtraction(Bs[c]-Ys[c],Ys[s]-Bs[s]))
    tmp -= np.mean(tmp)
    #tmp /= np.max(np.abs(tmp))
    #tmp *= 254
    #return (np.uint8(np.where(tmp>0,tmp,0)),np.uint8(np.where(tmp<0,-tmp,0)))
    return (np.float32(np.where(tmp>0,tmp,0)),np.float32(np.where(tmp<0,-tmp,0)))


##--------------------------##
"""
the following function is the most important function
"""
##--------------------------##

def Itti_down_sampling(resized_image, args, ifshow = False):
    #resized_image = np.uint8((np.float16(resized_image)/255) ** 0.218 * 255)
    # former_image= np.copy(resized_image)
    gamma = args.gamma_correlation
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
	# apply gamma correction using the lookup table
    resized_image =  cv2.LUT(resized_image, table)
    # resized_image = cv2.GaussianBlur(resized_image,(3,3),sigmaX=0.02*320,sigmaY=0.02*240)
    b, g, r = seperate_RGB_chanells(resized_image)
    
    # fig, (ax1,ax2) = plt.subplots(1,2)
    # ax1.imshow(cv2.cvtColor(former_image,cv2.COLOR_BGR2RGB))
    # ax1.set_title("without gamma correction")
    # ax2.imshow(cv2.cvtColor(resized_image,cv2.COLOR_BGR2RGB))
    # ax2.set_title("after gamma correction")
    # plt.show()

    long_edge_gaussian_kernal = cv2.getGaussianKernel(r.shape[0],r.shape[0]/2)
    short_edge_gaussian_kernal = cv2.getGaussianKernel(r.shape[1],r.shape[1]/2)
    long_edge_gaussian_kernal /= np.max(long_edge_gaussian_kernal)
    short_edge_gaussian_kernal /= np.max(short_edge_gaussian_kernal)
    # gaussian_kernel = long_edge_gaussian_kernal * short_edge_gaussian_kernal.T

    r, g, b = map(lambda x: long_edge_gaussian_kernal * x * short_edge_gaussian_kernal.T + (1-long_edge_gaussian_kernal * short_edge_gaussian_kernal.T) * np.mean(x), [r,g,b])
    # y = eight_pyrimid_built(np.uint8(np.abs(r//2+g//2 - np.abs(r//2 - g//2))))
    r = eight_pyrimid_built(r)
    g = eight_pyrimid_built(g)
    b = eight_pyrimid_built(b)
    
    
    # gaussian = eight_pyrimid_built(gaussian_kernel)
    # assert len(r) == len(g) == len(b)
    Is = [np.int16(0.299 * r[i] + 0.587 * g[i] + 0.114 * b[i]) for i in range(len(r))] # standard method of calculate Intensity.
    # Is = [np.uint8(r[i]/3 + g[i]/3 + b[i]/3) for i in range(len(r))]
    Ds = [np.max(Is[i]) - Is[i] for i in range(len(Is))]
    
    # maximum = max([np.max(Is[i]) for i in range(len(r))])


    # b = [np.where(b[i]>= 0.1 * maximum,b[i],0) for i in range(9)]
    # g = [np.where(g[i]>= 0.1 * maximum,g[i],0) for i in range(9)]
    # r = [np.where(r[i]>= 0.1 * maximum,r[i],0) for i in range(9)]
    
    rIs = [Is[i]-np.mean(Is[i]) for i in range(len(r))]
    rDs = [Ds[i]-np.mean(Ds[i]) for i in range(len(r))]
    Rs = [r[i]-(g[i]/2+b[i]/2) for i in range(len(r))]
    Gs = [g[i]-(r[i]/2+b[i]/2) for i in range(len(r))]
    Bs = [b[i]-(g[i]/2+r[i]/2) for i in range(len(r))]
    Ys = [(r[i]/2+g[i]/2) - np.abs(r[i]/2 - g[i]/2) - b[i] for i in range(len(r))]
    # Ys = [y[i] - b[i] for i in range(len(r))]


    
    rIs = [np.where(rI>0,rI,0) for rI in Is]
    rDs = [np.where(rD>0,rD,0) for rD in Ds]
    Rs = [np.where(R>0,R,0) for R in Rs]
    Gs = [np.where(G>0,G,0) for G in Gs]
    Bs = [np.where(B>0,B,0) for B in Bs]
    Ys = [np.where(Y>0,Y,0) for Y in Ys]
    
    if ifshow:
        plt.imshow(resized_image)
        plt.show()
        fig, axs = plt.subplots(6,len(Is))
        for i in range(len(Is)):
            axs[0,i].imshow(Is[i])
            axs[0,i].set_title("Intensity")
            axs[1,i].imshow(Ds[i])
            axs[1,i].set_title("Dark")
            axs[2,i].imshow(Rs[i])
            axs[2,i].set_title("Red")
            axs[3,i].imshow(Bs[i])
            axs[3,i].set_title("Blue")
            axs[4,i].imshow(Gs[i])
            axs[4,i].set_title("Green")
            axs[5,i].imshow(Ys[i])
            axs[5,i].set_title("Yellow")
        plt.show()

    return rIs, Rs, Gs, Bs, Ys, rDs, Is

# def Itti_down_sampling_PYROri(image, ifshow = False):
#     """
#     Only use one time Orientiation calculation, then filter
#     """
        
#     resized_image = resize_to_normal_shape(image)

#     b, g, r = seperate_RGB_chanells(resized_image)

#     r_sigma = eight_pyrimid_built(r)
#     g_sigma = eight_pyrimid_built(g)
#     b_sigma = eight_pyrimid_built(b)
#     I = [(r_sigma[i]/3+g_sigma[i]/3+b_sigma[i]/3) for i in range(9)]
#     maximum = [np.max(I[i]) for i in range(9)]


#     b = [np.where(b_sigma[i]>= 0.1 * maximum[i],b_sigma[i],0) for i in range(9)]
#     g = [np.where(g_sigma[i]>= 0.1 * maximum[i],g_sigma[i],0) for i in range(9)]
#     r = [np.where(r_sigma[i]>= 0.1 * maximum[i],r_sigma[i],0) for i in range(9)]

#     Is = I
#     Ds = [np.max(Is[0]) - single_I for single_I in Is]
#     Rs = [r[i]-(g[i]/2+b[i]/2) for i in range(9)]
#     Rs = [np.where(R>0,R,0) for R in Rs]
#     Gs = [g[i]-(r[i]/2+b[i]/2) for i in range(9)]
#     Gs = [np.where(G>0,G,0) for G in Gs]
#     Bs = [b[i]-(g[i]/2+r[i]/2) for i in range(9)]
#     Bs = [np.where(B>0,B,0) for B in Bs]
#     Ys = [(r[i]/2+g[i]/2) - np.abs(r[i]/2 - g[i]/2) - b[i] for i in range(9)]
#     Ys = [np.where(Y>0,Y,0) for Y in Ys]
#     if ifshow:
#         fig, ((ax1,ax2,ax3),(ax4,ax5,ax6)) = plt.subplots(2,3)
#         ax1.imshow(Is[0])
#         ax1.set_title("Intensity")
#         ax4.imshow(Ds[0])
#         ax4.set_title("Dark")
#         ax2.imshow(Rs[0])
#         ax2.set_title("Red")
#         ax3.imshow(Bs[0])
#         ax3.set_title("Blue")
#         ax5.imshow(Gs[0])
#         ax5.set_title("Green")
#         ax6.imshow(Ys[0])
#         ax6.set_title("Yellow")
#         plt.show()
#     return Is, Rs, Gs, Bs, Ys, Ds

def Itti_feature_maps(Is,Rs,Gs,Bs,Ys,Ds,norm_lib,args,ifshow=False):
    # c_set = (2,3,4)
    c_set = args.center
    delta_set = args.surrounding

    I_dict = {}
    RG_dict = {}
    BY_dict = {}
    color_max = 1
    
    for c in c_set:
        for delta in delta_set:
            I_dict[(c,c+delta)] = Is_scale(Is,Ds,c,c+delta)
            
            RG_dict[(c,c+delta)] = RG_scale(Rs,Gs,c,c+delta)
            # fig, axs = plt.subplots(2,3)
            # axs[0,0].imshow(Rs[c])
            # axs[1,0].imshow(Rs[c+delta])
            # axs[0,1].imshow(Gs[c])
            # axs[1,1].imshow(Gs[c+delta])
            # axs[0,2].imshow(RG_dict[(c,c+delta)][0])
            # axs[1,2].imshow(RG_dict[(c,c+delta)][1])
            # plt.show()
            # fig, axs = plt.subplots(1,2)
            # axs[0].imshow(normalize_img(RG_dict[(c,c+delta)][0],norm_lib))
            # axs[1].imshow(normalize_img(RG_dict[(c,c+delta)][1],norm_lib))
            # plt.show()
            BY_dict[(c,c+delta)] = BY_scale(Bs,Ys,c,c+delta)
    #         color_max = max(np.max(RG_dict[(c,c+delta)]),np.max(BY_dict[(c,c+delta)]),color_max)
    # for c in c_set:
    #     for delta in delta_set:
    #         RG_dict[(c,c+delta)] = (254 * RG_dict[(c,c+delta)][0]/color_max, 254 * RG_dict[(c,c+delta)][1]/color_max)
    #         BY_dict[(c,c+delta)] = (254 * BY_dict[(c,c+delta)][0]/color_max, 254 * BY_dict[(c,c+delta)][1]/color_max)
    if ifshow:
        fig,axs = plt.subplots(6,len(c_set)*len(delta_set))
        if len(c_set)*len(delta_set) == 1:
            plt.suptitle(f"{(c_set[0],c_set[0]+delta_set[0])}")
            axs[0].imshow(I:=normalize_img(I_dict[(c_set[0],c_set[0]+delta_set[0])][0],norm_lib))
            axs[1].imshow(D:=normalize_img(I_dict[(c_set[0],c_set[0]+delta_set[0])][1],norm_lib))
            axs[2].imshow(R:= normalize_img(RG_dict[(c_set[0],c_set[0]+delta_set[0])][0],norm_lib))
            axs[3].imshow(G:=normalize_img(RG_dict[(c_set[0],c_set[0]+delta_set[0])][1],norm_lib))
            axs[4].imshow(B:=normalize_img(BY_dict[(c_set[0],c_set[0]+delta_set[0])][0],norm_lib))
            axs[5].imshow(Y:=normalize_img(BY_dict[(c_set[0],c_set[0]+delta_set[0])][1],norm_lib))
            axs[0].set_title(f"I max = {np.max(I)}")
            axs[1].set_title(f"D max = {np.max(D)}")
            axs[2].set_title(f"R max = {np.max(R)}")
            axs[3].set_title(f"G max = {np.max(G)}")
            axs[4].set_title(f"B max = {np.max(B)}")
            axs[5].set_title(f"Y max = {np.max(Y)}")
            
        else:
            for ic, c in enumerate(c_set):
                for id, d in enumerate(delta_set):
                    axs[0,ic*len(delta_set) + id].imshow(normalize_img(I_dict[(c,c+d)][0],norm_lib))
                    axs[0,ic*len(delta_set) + id].set_title({(c,c+d)})
                    axs[1,ic*len(delta_set) + id].imshow(normalize_img(I_dict[(c,c+d)][1],norm_lib))
                    axs[2,ic*len(delta_set) + id].imshow(normalize_img(RG_dict[(c,c+d)][0],norm_lib))
                    axs[3,ic*len(delta_set) + id].imshow(normalize_img(RG_dict[(c,c+d)][1],norm_lib))
                    axs[4,ic*len(delta_set) + id].imshow(normalize_img(BY_dict[(c,c+d)][0],norm_lib))
                    axs[5,ic*len(delta_set) + id].imshow(normalize_img(BY_dict[(c,c+d)][1],norm_lib))
        plt.show()

    return I_dict, RG_dict, BY_dict


def synthesis_conspicuous_map(I_dict, RG_dict, BY_dict,norm_lib,args,ifshow=False):
    c_set = args.center
    delta_set = args.surrounding
    I_bar = np.zeros((1,1))
    C_bar = np.zeros((1,1))
    if not (np.any(I_dict[(c_set[0],c_set[0]+delta_set[0])][0] > 0) or np.any(I_dict[(c_set[0],c_set[0]+delta_set[0])][1] > 0)):
        long_edge_gaussian_kernal = cv2.getGaussianKernel(I_dict[(c_set[0],c_set[0]+delta_set[0])][0].shape[0],I_dict[(c_set[0],c_set[0]+delta_set[0])][0].shape[1]/3)
        short_edge_gaussian_kernal = cv2.getGaussianKernel(I_dict[(c_set[0],c_set[0]+delta_set[0])][0].shape[1],I_dict[(c_set[0],c_set[0]+delta_set[0])][0].shape[1]/3)
        center_img = long_edge_gaussian_kernal * short_edge_gaussian_kernal.T
        center_img /= np.max(center_img)
        center_img = np.float32(center_img)
        return center_img, center_img
    for c in c_set:
        for delta in delta_set:
            assert I_dict[(c,c+delta)][0].shape == I_dict[(c,c+delta)][1].shape == RG_dict[(c,c+delta)][0].shape == RG_dict[(c,c+delta)][1].shape == BY_dict[(c,c+delta)][0].shape == BY_dict[(c,c+delta)][1].shape
            #long_edge_gaussian_kernal = cv2.getGaussianKernel(I_dict[(c,c+delta)][0].shape[0],I_dict[(c,c+delta)][0].shape[1]/3)
            #short_edge_gaussian_kernal = cv2.getGaussianKernel(I_dict[(c,c+delta)][0].shape[1],I_dict[(c,c+delta)][0].shape[1]/3)
            #long_edge_gaussian_kernal /= np.max(long_edge_gaussian_kernal)
            #short_edge_gaussian_kernal /= np.max(short_edge_gaussian_kernal)

            # plt.imshow(long_edge_gaussian_kernal * short_edge_gaussian_kernal.T)
            # plt.show()
            # plt.imshow(I_dict[(c,c+delta)][0])
            # plt.show()
            # plt.imshow(I_dict[(c,c+delta)][1])
            # plt.show()

            I_bar = addition(I_bar, normalize_img(I_dict[(c,c+delta)][0],norm_lib))  # White
            I_bar = addition(I_bar, normalize_img(I_dict[(c,c+delta)][1],norm_lib))  # Black
            C_bar = addition(C_bar, normalize_img(RG_dict[(c,c+delta)][0],norm_lib)) # Red
            C_bar = addition(C_bar, normalize_img(RG_dict[(c,c+delta)][1],norm_lib)) # Green
            C_bar = addition(C_bar, normalize_img(BY_dict[(c,c+delta)][0],norm_lib)) # Blue
            C_bar = addition(C_bar, normalize_img(BY_dict[(c,c+delta)][1],norm_lib)) # Yellow
            #I_bar = addition(I_bar,long_edge_gaussian_kernal * normalize_img(I_dict[(c,c+delta)][0],norm_lib) * short_edge_gaussian_kernal.T)  # White
            #I_bar = addition(I_bar,long_edge_gaussian_kernal * normalize_img(I_dict[(c,c+delta)][1],norm_lib) * short_edge_gaussian_kernal.T)  # Black
            #C_bar = addition(C_bar,long_edge_gaussian_kernal * normalize_img(RG_dict[(c,c+delta)][0],norm_lib) * short_edge_gaussian_kernal.T) # Red
            #C_bar = addition(C_bar,long_edge_gaussian_kernal * normalize_img(RG_dict[(c,c+delta)][1],norm_lib) * short_edge_gaussian_kernal.T) # Green
            #C_bar = addition(C_bar,long_edge_gaussian_kernal * normalize_img(BY_dict[(c,c+delta)][0],norm_lib) * short_edge_gaussian_kernal.T) # Blue
            #C_bar = addition(C_bar,long_edge_gaussian_kernal * normalize_img(BY_dict[(c,c+delta)][1],norm_lib) * short_edge_gaussian_kernal.T) # Yellow
            if ifshow:
                # show all adding part:
                fig, ((ax1,ax2,ax3,ax7),(ax4,ax5,ax6,ax8)) = plt.subplots(2,4)
                #white = long_edge_gaussian_kernal * normalize_img(I_dict[(c,c+delta)][0],norm_lib) * short_edge_gaussian_kernal.T
                white = normalize_img(I_dict[(c,c+delta)][0],norm_lib)
                ax1.imshow(white)
                ax1.set_title(f"White,max = {np.max(white)}")
                #black = long_edge_gaussian_kernal * normalize_img(I_dict[(c,c+delta)][1],norm_lib) * short_edge_gaussian_kernal.T
                ax4.imshow(black:=normalize_img(I_dict[(c,c+delta)][1],norm_lib) )
                ax4.set_title(f"Black,max = {np.max(black)}")
                #red = long_edge_gaussian_kernal * normalize_img(RG_dict[(c,c+delta)][0],norm_lib)* short_edge_gaussian_kernal.T
                ax2.imshow(red:=normalize_img(RG_dict[(c,c+delta)][0],norm_lib) )
                ax2.set_title(f"Red,max = {np.max(red)}")
                #green = long_edge_gaussian_kernal * normalize_img(RG_dict[(c,c+delta)][1],norm_lib)* short_edge_gaussian_kernal.T
                ax5.imshow(green:=normalize_img(RG_dict[(c,c+delta)][1],norm_lib) )
                ax5.set_title(f"Green,max = {np.max(green)}")
                #blue = long_edge_gaussian_kernal * normalize_img(BY_dict[(c,c+delta)][0],norm_lib)* short_edge_gaussian_kernal.T
                ax3.imshow(blue:=normalize_img(BY_dict[(c,c+delta)][0],norm_lib) )
                ax3.set_title(f"Blue,max = {np.max(blue)}")
                #yellow = long_edge_gaussian_kernal * normalize_img(BY_dict[(c,c+delta)][1],norm_lib)* short_edge_gaussian_kernal.T
                ax6.imshow(yellow:=normalize_img(BY_dict[(c,c+delta)][1],norm_lib) )
                ax6.set_title(f"Yellow,max = {np.max(yellow)}")
                # plt.show()
                ax7.imshow(I_bar)
                ax8.imshow(C_bar)
                plt.show()
            
    return I_bar, C_bar


def single_image_conspicucous_map(image,norm_lib,args):
    """
    using single image to generate conspicuous maps
    """
    pIs,pRs,pGs,pBs,pYs,pDs = Itti_down_sampling(image,args)
    I_dict, RG_dict, BY_dict =Itti_feature_maps(pIs,pRs,pGs,pBs,pYs,pDs,norm_lib,args)
    I_bar, C_bar  = synthesis_conspicuous_map(I_dict, RG_dict, BY_dict,norm_lib,args)
    I_bar, C_bar  = normalize_img(I_bar,norm_lib), normalize_img(C_bar,norm_lib)
    return I_bar, C_bar


def parse_args():
    """
    get args.
    """
    parse = argparse.ArgumentParser(description='essential parameters') 
    parse.add_argument('--video_path', default="E:\\Li Lab\\itti_and_lif\\RealtimeSaliency\\test_video\\492.avi", type=str, help='path of sample video') 
    parse.add_argument('--generate_name',default = "result_video\\492.mp4",type=str,help = 'default generate name')
    parse.add_argument('--total_height',default=9,type=int,help='total height of orientation pyramid, equals to height of gaussian pyramid + kernel size Pyramid,Itti default is 9')
    parse.add_argument('--pyramid_height', default=5, type=int, help='height of Gaussian Pyramid, tried maximum param is 5')
    # parse.add_argument('--gabor_kernel_size',default=33,type=int,help='the minimal value of gabor kernel size. when meet some constrains, we would double some params.')
    parse.add_argument('--gabor_kernel_size',default=9,type=int,help='the minimal value of gabor kernel size. when meet some constrains, we would double some params.')
    parse.add_argument('--num_of_thetas',default=4,type=int,help='different gabor filter orientations.')
    parse.add_argument('--mini_sigma',default=0.5,type=float,help='sigma of gabor kernel, if the image is too small hori2then double it.')
    parse.add_argument('--gabor_lambda',default=np.pi/np.sqrt(2*np.log(1/0.5)),type=float,help = 'lambda for gabor kernels')
    parse.add_argument('--gabor_gamma',default=1,type=float,help = 'gamma value for gabor filter.')
    parse.add_argument('--default_size',default=(320,240),type=tuple,help = 'default size of image')
    parse.add_argument('--control_length',default=16 ,type=int,help = 'default length of the control list')
    parse.add_argument('--default_checkpoint', default=(2,4,8,16),type=lambda s:tuple(int(item) for item in s.split(',')),help = 'default checkpoints of control length')
    parse.add_argument('--selective_threshold',default =00,type=float,help = 'default selective threshold percentage')
    parse.add_argument('--double_height',default = False,type=bool,help = 'default output window')
    parse.add_argument('--alpha_beta',default = (0.3,0.3),type=lambda s:tuple(float(item) for item in s.split(',')),help = 'determine standard alpha and beta.')
    parse.add_argument('--output',default="quad",type = str, help = "determine output format.")
    parse.add_argument('--target_folder',default='lower_result\\tests',type=str,help = 'default place to place pred results')
    parse.add_argument('--center',default = (1,2,3),type=lambda s:tuple(int(item) for item in s.split(',')[:-1]),help = "center params. Itti default params are (2, 3, 4), equals to (1,2,3) here, in fast paras they use (0,)")
    parse.add_argument('--surrounding',default = (3,4),type=lambda s:tuple(int(item) for item in s.split(',')[:-1]),help = "surrounding params. Itti default params are (3, 4),in fast pami they use (4,)")
    parse.add_argument('--motion_center',default = (1,2,3),type=lambda s:tuple(int(item) for item in s.split(',')),help = "center params. Itti default params are (2, 3, 4)")
    parse.add_argument('--motion_surrounding',default = (3,4),type=lambda s:tuple(int(item) for item in s.split(',')),help = "surrounding params. Itti default params are (3, 4)")
    parse.add_argument('--decay_factor',default = 0.9,type=float,help = "decay factor of time-related problems.")
    parse.add_argument('--gamma_correlation',default=1,type = float, help = "gamma value for gamma correlation")
    parse.add_argument('--tmp_save_factor',default=0.9,type = float, help = "gamma value for gamma correlation")
    parse.add_argument('--continuity',type=bool, default=True,help="control how to output, using simple output or continue output.")
    parse.add_argument('--add_central_gaussian',type = bool, default= True, help = "control if apply central gaussian in the end")
    parse.add_argument('--output_gamma_correlation',type = bool, default= False, help = "control if apply central gaussian in the end")
    parse.add_argument('--generate_type',type = str, default= 'both', help = "static, dynamic or both.")
    args = parse.parse_args() 
    return args

def main(image,args):
    ifshow = False
    rIs, Rs, Gs, Bs, Ys, rDs, Is = Itti_down_sampling(image,args,ifshow)
    # fig,((ax11,ax12,ax13,ax14,ax15,ax16,ax17,ax18,ax19),\
    #      (ax21,ax22,ax23,ax24,ax25,ax26,ax27,ax28,ax29),\
    #      (ax31,ax32,ax33,ax34,ax35,ax36,ax37,ax38,ax39),\
    #      (ax41,ax42,ax43,ax44,ax45,ax46,ax47,ax48,ax49),\
    #      (ax51,ax52,ax53,ax54,ax55,ax56,ax57,ax58,ax59),\
    #      (ax61,ax62,ax63,ax64,ax65,ax66,ax67,ax68,ax69)) = plt.subplots(6,9)
    # full_lst = [[],pIs,pDs,pRs,pGs,pBs,pYs]
    # for img_index in range(1,7):
    #     for scale_index in range(1,10):
    #         exec(f"ax{img_index}{scale_index}.imshow(full_lst[{img_index}][{scale_index-1}])")
    # plt.show()

    # pIs,pRs,pGs,pBs,pYs,pDs = Itti_down_sampling(image)
    # fig,((ax11,ax12,ax13,ax14,ax15,ax16,ax17,ax18,ax19),\
    #      (ax21,ax22,ax23,ax24,ax25,ax26,ax27,ax28,ax29),\
    #      (ax31,ax32,ax33,ax34,ax35,ax36,ax37,ax38,ax39),\
    #      (ax41,ax42,ax43,ax44,ax45,ax46,ax47,ax48,ax49),\
    #      (ax51,ax52,ax53,ax54,ax55,ax56,ax57,ax58,ax59)) = plt.subplots(5,9)
    # full_lst = [[],pIs,pRs,pGs,pBs,pYs]
    # for img_index in range(1,6):
    #     for scale_index in range(1,10):
    #         exec(f"ax{img_index}{scale_index}.imshow(full_lst[{img_index}][{scale_index-1}])")
    # plt.show()

    # fig,((ax11,ax12,ax13,ax14,ax15,ax16,ax17,ax18,ax19),\
    #      (ax21,ax22,ax23,ax24,ax25,ax26,ax27,ax28,ax29),\
    #      (ax31,ax32,ax33,ax34,ax35,ax36,ax37,ax38,ax39),\
    #      (ax41,ax42,ax43,ax44,ax45,ax46,ax47,ax48,ax49)) = plt.subplots(4,9)

    # for degree_index in range(0,4):
    #     for scale_index in range(1,10):
    #         exec(f"ax{degree_index+1}{scale_index}.imshow(pOs[{scale_index-1}][{degree_index}])")
    # plt.show()


    I_dict, RG_dict, BY_dict = Itti_feature_maps(rIs, Rs, Gs, Bs, Ys, rDs, norm_lib, args,ifshow)
    I_bar, C_bar = synthesis_conspicuous_map(I_dict, RG_dict, BY_dict,norm_lib,args,ifshow)

    O_bar = normalize_img(O_bar_synthesis(Is.copy(),args,gabor_lib,norm_lib),norm_lib)
    static_saliency_map = addition(I_bar/(np.sum(I_bar)+1e-3)+ C_bar/(np.sum(C_bar)+1e-3), (O_bar/np.sum(O_bar)+1e-3))

    I_bar, C_bar = map(lambda x: normalize_img(x,norm_lib),[I_bar, C_bar])
    image_saliency = (I_bar + C_bar + O_bar) / 3
    Saliency_map = normalize_img(image_saliency,norm_lib)
    Saliency_map = 254 * Saliency_map / np.max(Saliency_map) if  np.max(Saliency_map) != 0 else Saliency_map
    Saliency_map = np.uint8(Saliency_map)
    Saliency_map = cv2.resize(Saliency_map,(image.shape[1],image.shape[0]),interpolation=cv2.INTER_NEAREST)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    Saliency_map = cv2.cvtColor(Saliency_map,cv2.COLOR_GRAY2RGB)

    fig, axs = plt.subplots(2,3)
    axs[0,0].imshow(I_bar)
    axs[0,0].set_title("I_bar")
    axs[0,1].imshow(C_bar)
    axs[0,1].set_title("C_bar")
    axs[0,2].imshow(image)
    axs[0,2].set_title("original image")
    axs[1,0].imshow(O_bar)
    axs[1,0].set_title("O_bar")
    axs[1,1].imshow(Saliency_map)
    axs[1,1].set_title("Saliency_map")
    axs[1,2].imshow((Saliency_map//8*7+image//8))
    axs[1,2].set_title("saliency map on image")
    plt.title(f"image for c_set = {args.center}, $\\delta$ set = {args.surrounding}")
    plt.show()

if __name__ == "__main__":
    args = parse_args()
    gabor_lib = initialize_gabor_lib()
    norm_lib = load_maximum_dll()
    """
    test code
    """
    # image = cv2.imread("e:\\Li Lab\\itti_and_lif\\test_jpgs\\bar.jpg")
    path = 'C:\\Users\\10690\\Pictures\\sample_image.jpg'
    main(cv2.imread(path),args)
    quit()
    path = 'E:\\Li Lab\\itti_and_lif\\self_designed_test_image\\re_design'#"e:\\Li Lab\\itti_and_lif\\self_designed_test_image\\design\\tmp_save"
    image_names = os.listdir(path)
    image_names = [os.path.join(path, img_name) for img_name in image_names if img_name.endswith(".JPG")]
    for img_name in image_names:
        #if img_name.find("green") == -1:
        #    continue
        image = cv2.imread(os.path.join(path,img_name))
        image = resize_to_normal_shape(image)
        main(image,args)
    #pIs,pRs,pGs,pBs,pYs,pOs,pDs = Itti_down_sampling(image)
    