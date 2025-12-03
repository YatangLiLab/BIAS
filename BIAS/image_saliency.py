import cv2
import matplotlib.pyplot as plt
import numpy as np

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
    # resized_image = cv2.resize(image,(640,480),interpolation=cv2.INTER_LINEAR)
    # resized_image = cv2.resize(image,(640,480),interpolation=cv2.INTER_NEAREST)
    resized_image = cv2.resize(image,(320,240),interpolation=cv2.INTER_NEAREST)
    return resized_image

def seperate_RGB_chanells(image):
   """
    Seperate RGB chanells to going through following algorithms
    """
   return image[:,:,0], image[:,:,1], image[:,:,2]


def Is_scale(img_lst,Darklist,c,s):
    tmp = subtraction(img_lst[c] - Darklist[c], Darklist[s] - img_lst[s])
    tmp -= np.mean(tmp)
    #tmp /= np.max(np.abs(tmp))
    #tmp *= 254
    #plt.imshow(tmp)
    #plt.title(f"Is scale{c},{s}")
    #plt.show()
    return (np.float32(np.where(tmp>0,tmp,0)),np.float32(np.where(tmp<0,-tmp,0)))
    #tmp = subtraction(img_lst[c],img_lst[s])
    #return (np.int16(np.where(tmp>0,tmp,0)),np.int16(np.where(tmp<0,-tmp,0)))

def RG_scale(Rs,Gs,c,s):
    tmp = subtraction(Rs[c]-Gs[c],Gs[s]-Rs[s])
    tmp -= np.mean(tmp)
    #tmp /= np.max(np.abs(tmp))
    #tmp *= 254
    #return (np.uint8(np.where(tmp>0,tmp,0)),np.uint8(np.where(tmp<0,-tmp,0)))
    return (np.float32(np.where(tmp>0,tmp,0)),np.float32(np.where(tmp<0,-tmp,0)))

def BY_scale(Bs,Ys,c,s):
    tmp = subtraction(Bs[c]-Ys[c],Ys[s]-Bs[s])
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

def Itti_down_sampling(resized_image, args, ifshow = True):
    #resized_image = np.uint8((np.float16(resized_image)/255) ** 0.218 * 255)
    # former_image= np.copy(resized_image)
    gamma = args.gamma_correlation
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
	# apply gamma correction using the lookup table
    resized_image =  cv2.LUT(resized_image, table)
    resized_image = cv2.GaussianBlur(resized_image,(3,3),sigmaX=0.02*320,sigmaY=0.02*240)
    b, g, r = seperate_RGB_chanells(resized_image)
    
    # fig, (ax1,ax2) = plt.subplots(1,2)
    # ax1.imshow(cv2.cvtColor(former_image,cv2.COLOR_BGR2RGB))
    # ax1.set_title("without gamma correction")
    # ax2.imshow(cv2.cvtColor(resized_image,cv2.COLOR_BGR2RGB))
    # ax2.set_title("after gamma correction")
    # plt.show()
    #long_edge_gaussian_kernal = cv2.getGaussianKernel(r.shape[0],r.shape[0]/1)
    #short_edge_gaussian_kernal = cv2.getGaussianKernel(r.shape[1],r.shape[1]/1)
    #long_edge_gaussian_kernal /= np.max(long_edge_gaussian_kernal)
    #short_edge_gaussian_kernal /= np.max(short_edge_gaussian_kernal)
    #gaussian_kernel = long_edge_gaussian_kernal * short_edge_gaussian_kernal.T

    # r, g, b = map(lambda x: long_edge_gaussian_kernal * x * short_edge_gaussian_kernal.T, [r,g,b])
    r = eight_pyrimid_built(r)
    g = eight_pyrimid_built(g)
    b = eight_pyrimid_built(b)
    # gaussian = eight_pyrimid_built(gaussian_kernel)
    # assert len(r) == len(g) == len(b)
    Is = [(0.299 * r[i] + 0.587 * g[i] + 0.114 * b[i]) for i in range(len(r))] # standard method of calculate Intensity.
    Ds = [cv2.normalize(np.max(Is[i]) - Is[i],None,0,255,cv2.NORM_MINMAX) for i in range(len(Is))]
    
    maximum = [np.max(Is[i]) for i in range(len(r))]


    #b = [np.where(b_sigma[i]>= 0.1 * maximum[i],b_sigma[i],0) for i in range(9)]
    #g = [np.where(g_sigma[i]>= 0.1 * maximum[i],g_sigma[i],0) for i in range(9)]
    #r = [np.where(r_sigma[i]>= 0.1 * maximum[i],r_sigma[i],0) for i in range(9)]
    threshold = max(maximum)
    
    rIs = [Is[i]-np.mean(Is[i]) for i in range(len(r))]
    rDs = [Ds[i]-np.mean(Ds[i]) for i in range(len(r))]
    Rs = [r[i]-(g[i]/2+b[i]/2) for i in range(len(r))]
    Gs = [g[i]-(r[i]/2+b[i]/2) for i in range(len(r))]
    Bs = [b[i]-(g[i]/2+r[i]/2) for i in range(len(r))]
    Ys = [(r[i]/2+g[i]/2) - np.abs(r[i]/2 - g[i]/2) - b[i] for i in range(len(r))]
    
    rIs = [np.where(rI>0.1 * threshold,rI,0) for rI in rIs]
    rDs = [np.where(rD>0.1 * threshold,rD,0) for rD in rDs]
    Rs = [np.where(R>0.1 * threshold,R,0) for R in Rs]
    Gs = [np.where(G>0.1 * threshold,G,0) for G in Gs]
    Bs = [np.where(B>0.1 * threshold,B,0) for B in Bs]
    Ys = [np.where(Y>0.1 * threshold,Y,0) for Y in Ys]

    if ifshow:
        fig, ((ax1,ax2,ax3),(ax4,ax5,ax6)) = plt.subplots(2,3)
        ax1.imshow(rIs[0])
        ax1.set_title("Intensity")
        ax4.imshow(rDs[0])
        ax4.set_title("Dark")
        ax2.imshow(Rs[0])
        ax2.set_title("Red")
        ax3.imshow(Bs[0])
        ax3.set_title("Blue")
        ax5.imshow(Gs[0])
        ax5.set_title("Green")
        ax6.imshow(Ys[0])
        ax6.set_title("Yellow")
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

def Itti_feature_maps(Is,Rs,Gs,Bs,Ys,Ds,norm_lib,args,ifshow=True):
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
            BY_dict[(c,c+delta)] = BY_scale(Bs,Ys,c,c+delta)
            color_max = max(np.max(RG_dict[(c,c+delta)]),np.max(BY_dict[(c,c+delta)]),color_max)
    for c in c_set:
        for delta in delta_set:
            RG_dict[(c,c+delta)] = (254 * RG_dict[(c,c+delta)][0]/color_max, 254 * RG_dict[(c,c+delta)][1]/color_max)
            BY_dict[(c,c+delta)] = (254 * BY_dict[(c,c+delta)][0]/color_max, 254 * BY_dict[(c,c+delta)][1]/color_max)
    if ifshow:
        fig,axs = plt.subplots(6,len(c_set)*len(delta_set))
        if len(c_set)*len(delta_set) == 1:
            axs[0].imshow(normalize_img(I_dict[(c_set[0],c_set[0]+delta_set[0])][0],norm_lib))
            axs[0].set_title(f"{(c_set[0],c_set[0]+delta_set[0])}")
            axs[1].imshow(normalize_img(I_dict[(c_set[0],c_set[0]+delta_set[0])][1],norm_lib))
            axs[2].imshow(normalize_img(RG_dict[(c_set[0],c_set[0]+delta_set[0])][0],norm_lib))
            axs[3].imshow(normalize_img(RG_dict[(c_set[0],c_set[0]+delta_set[0])][1],norm_lib))
            axs[4].imshow(normalize_img(BY_dict[(c_set[0],c_set[0]+delta_set[0])][0],norm_lib))
            axs[5].imshow(normalize_img(BY_dict[(c_set[0],c_set[0]+delta_set[0])][1],norm_lib))
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


def synthesis_conspicuous_map(I_dict, RG_dict, BY_dict,norm_lib,args,ifshow=True):
    c_set = args.center
    delta_set = args.surrounding
    I_bar = np.zeros((1,1))
    C_bar = np.zeros((1,1))
    for c in c_set:
        for delta in delta_set:
            assert I_dict[(c,c+delta)][0].shape == I_dict[(c,c+delta)][1].shape == RG_dict[(c,c+delta)][0].shape == RG_dict[(c,c+delta)][1].shape == BY_dict[(c,c+delta)][0].shape == BY_dict[(c,c+delta)][1].shape
            #long_edge_gaussian_kernal = cv2.getGaussianKernel(I_dict[(c,c+delta)][0].shape[0],I_dict[(c,c+delta)][0].shape[1]/3)
            #short_edge_gaussian_kernal = cv2.getGaussianKernel(I_dict[(c,c+delta)][0].shape[1],I_dict[(c,c+delta)][0].shape[1]/3)
            #long_edge_gaussian_kernal /= np.max(long_edge_gaussian_kernal)
            #short_edge_gaussian_kernal /= np.max(short_edge_gaussian_kernal)

            # plt.imshow(long_edge_gaussian_kernal * short_edge_gaussian_kernal.T)
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
                fig, ((ax1,ax2,ax3),(ax4,ax5,ax6)) = plt.subplots(2,3)
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
                plt.show()
                # plt.imshow(I_bar)
                # plt.show()
                # plt.imshow(C_bar)
                # plt.show()
            
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

if __name__ == "__main__":
    """
    test code
    """
    image = cv2.imread("")
    #pIs,pRs,pGs,pBs,pYs,pOs,pDs = Itti_down_sampling(image)
    pIs,pRs,pGs,pBs,pYs,pDs = Itti_down_sampling(image)
    fig,((ax11,ax12,ax13,ax14,ax15,ax16,ax17,ax18,ax19),\
         (ax21,ax22,ax23,ax24,ax25,ax26,ax27,ax28,ax29),\
         (ax31,ax32,ax33,ax34,ax35,ax36,ax37,ax38,ax39),\
         (ax41,ax42,ax43,ax44,ax45,ax46,ax47,ax48,ax49),\
         (ax51,ax52,ax53,ax54,ax55,ax56,ax57,ax58,ax59),\
         (ax61,ax62,ax63,ax64,ax65,ax66,ax67,ax68,ax69)) = plt.subplots(6,9)
    full_lst = [[],pIs,pDs,pRs,pGs,pBs,pYs]
    for img_index in range(1,7):
        for scale_index in range(1,10):
            exec(f"ax{img_index}{scale_index}.imshow(full_lst[{img_index}][{scale_index-1}])")
    plt.show()

    pIs,pRs,pGs,pBs,pYs,pDs = Itti_down_sampling(image)
    fig,((ax11,ax12,ax13,ax14,ax15,ax16,ax17,ax18,ax19),\
         (ax21,ax22,ax23,ax24,ax25,ax26,ax27,ax28,ax29),\
         (ax31,ax32,ax33,ax34,ax35,ax36,ax37,ax38,ax39),\
         (ax41,ax42,ax43,ax44,ax45,ax46,ax47,ax48,ax49),\
         (ax51,ax52,ax53,ax54,ax55,ax56,ax57,ax58,ax59)) = plt.subplots(5,9)
    full_lst = [[],pIs,pRs,pGs,pBs,pYs]
    for img_index in range(1,6):
        for scale_index in range(1,10):
            exec(f"ax{img_index}{scale_index}.imshow(full_lst[{img_index}][{scale_index-1}])")
    plt.show()

    # fig,((ax11,ax12,ax13,ax14,ax15,ax16,ax17,ax18,ax19),\
    #      (ax21,ax22,ax23,ax24,ax25,ax26,ax27,ax28,ax29),\
    #      (ax31,ax32,ax33,ax34,ax35,ax36,ax37,ax38,ax39),\
    #      (ax41,ax42,ax43,ax44,ax45,ax46,ax47,ax48,ax49)) = plt.subplots(4,9)

    # for degree_index in range(0,4):
    #     for scale_index in range(1,10):
    #         exec(f"ax{degree_index+1}{scale_index}.imshow(pOs[{scale_index-1}][{degree_index}])")
    # plt.show()


    I_dict, RG_dict, BY_dict =Itti_feature_maps(pIs,pRs,pGs,pBs,pYs,pDs)
    I_bar, C_bar = synthesis_conspicuous_map(I_dict, RG_dict, BY_dict,(pIs[4].shape[1],pIs[4].shape[0]))
    plt.imshow(I_bar)
    plt.title("Ibar")
    plt.show()
    plt.imshow(C_bar)
    plt.title("Cbar")
    plt.show()
    I_bar, C_bar = map(normalize_img,[I_bar, C_bar])
    image_saliency = (I_bar + C_bar ) / 2
    Saliency_map = normalize_img(image_saliency)
    Saliency_map = 254 * Saliency_map / np.max(Saliency_map)
    Saliency_map = np.uint8(Saliency_map)
    Saliency_map = cv2.resize(Saliency_map,(image.shape[1],image.shape[0]),interpolation=cv2.INTER_NEAREST)
    plt.imshow(Saliency_map)
    plt.title("Saliency Map")
    plt.show()