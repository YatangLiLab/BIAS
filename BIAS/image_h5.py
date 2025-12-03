import cv2
import argparse
import tqdm
import time
import numpy as np
import matplotlib.pyplot as plt
from utils import read_video, normalize_img, addition,interference_function, initialize_gabor_lib,load_maximum_dll,multation
from newStaticSaliency import resize_to_normal_shape, Itti_down_sampling, Itti_feature_maps, synthesis_conspicuous_map
from selfMovement import four_dir_sim, Intensity_processing, Camera
from orientation_saliency import O_bar_synthesis
import os
import h5py

def parse_args():
    """
    get args.
    """
    parse = argparse.ArgumentParser(description='essential parameters') 
    parse.add_argument('--video_path', default="", type=str, help='path of sample video') 
    parse.add_argument('--generate_name',default = "",type=str,help = 'default generate name')
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
    parse.add_argument('--alpha_beta',default = (1.0,1.0),type=lambda s:tuple(float(item) for item in s.split(',')),help = 'determine standard alpha and beta.')
    parse.add_argument('--output',default="quad",type = str, help = "determine output format.")
    parse.add_argument('--target_folder',default='lower_result\\tests',type=str,help = 'default place to place pred results')
    parse.add_argument('--center',default = (1,),type=lambda s:tuple(int(item) for item in s.split(',')[:-1]),help = "center params. Itti default params are (2, 3, 4), equals to (1,2,3) here, in fast paras they use (0,)")
    parse.add_argument('--surrounding',default = (4,),type=lambda s:tuple(int(item) for item in s.split(',')[:-1]),help = "surrounding params. Itti default params are (3, 4),in fast pami they use (4,)")
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

def main(args:argparse.Namespace):
    """
    the main function of dynamic scene saliency map prediction.
    """
    #------------------------------------------------initialize---------------------------------------------------------
    gabor_lib = initialize_gabor_lib()
    norm_lib = load_maximum_dll()
    alpha, beta = args.alpha_beta
    video_path = args.video_path
    video_cap, final_shape = read_video(video_path)
    ret, first_frame = video_cap.read()
    if not ret:
        # check if loaded correctly
        raise RuntimeError(f"failed to load video! Check video path = {video_path}")
    frame_count = 0 # remember howmany images we have processed
    height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_number = video_cap.get(cv2.CAP_PROP_FRAME_COUNT) # amount of frame
    if args.double_height:
        # see if we want to generate shown video
        final_shape = (width,height*2)
    tmp_save_factor = args.tmp_save_factor
    #------------------------------------------------FirstImageProcessing---------------------------------------------------------

    # first we calculate the full gaussian pyramid.
    #resized_image = resize_to_normal_shape(first_frame)
    resized_image = cv2.resize(first_frame,(320,240),interpolation=cv2.INTER_NEAREST)
    rIs, Rs, Gs, Bs, Ys, rDs, Is = Itti_down_sampling(resized_image,args) # intensity channels and color channels

    image_saliency_map = np.zeros((height,width,int(frame_number)),dtype=np.float16)

    # calculate first image only based on static methods.
    I_dict, RG_dict, BY_dict = Itti_feature_maps(rIs, Rs, Gs, Bs, Ys, rDs, norm_lib, args)
    I_bar, C_bar = synthesis_conspicuous_map(I_dict, RG_dict, BY_dict,norm_lib,args)
    I_bar, C_bar = normalize_img(I_bar,norm_lib),normalize_img(C_bar,norm_lib)
    O_bar = normalize_img(O_bar_synthesis(Is.copy(),args,gabor_lib,norm_lib),norm_lib)
    static_saliency_map = addition(I_bar/(np.sum(I_bar)+1e-3)+ C_bar/(np.sum(C_bar)+1e-3), (O_bar/np.sum(O_bar)+1e-3))
    static_saliency_map = normalize_img(static_saliency_map,norm_lib)
    static_saliency_map = cv2.resize(static_saliency_map, (width,height))
    image_saliency_map[:,:,0] =np.float16(static_saliency_map)
    #static_saliency_map = cv2.normalize(static_saliency_map, None, 0, 255, cv2.NORM_MINMAX) # minmax coding
    #------------------------------------------------VideoProcessing---------------------------------------------------------

    for _count in tqdm.trange(1,int(frame_number)): # in range(int(frame_number)): #
        ret, frame = video_cap.read()
        if not ret: # whole video is processed
            print("Done!")
            break
        #time0 = time.time()
        resized_image = cv2.resize(frame,(320,240),interpolation=cv2.INTER_NEAREST)# resize_to_normal_shape(frame)

        rIs, Rs, Gs, Bs, Ys, rDs, Is = Itti_down_sampling(resized_image,args) # intensity channels and color channels
        #time01 = time.time() 
        
        # DownSampleing Pyramids
        
        I_dict, RG_dict, BY_dict = Itti_feature_maps(rIs, Rs, Gs, Bs, Ys, rDs,norm_lib ,args)
        I_bar, C_bar = synthesis_conspicuous_map(I_dict, RG_dict, BY_dict,norm_lib,args)
        I_bar, C_bar = synthesis_conspicuous_map(I_dict, RG_dict, BY_dict,norm_lib,args)
        #time05 = time.time()
        #generate Intensity and color Saliency map


        O_bar = normalize_img(O_bar_synthesis(Is.copy(),args,gabor_lib,norm_lib),norm_lib)
        static_saliency_map = addition(I_bar/(np.sum(I_bar)+1e-3)+ C_bar/(np.sum(C_bar)+1e-3), (O_bar/np.sum(O_bar)+1e-3))
        norm_static = normalize_img(static_saliency_map,norm_lib)

        if args.add_central_gaussian:
            long_edge_gaussian_kernal = cv2.getGaussianKernel(norm_static.shape[0],norm_static.shape[0]/2)
            long_edge_gaussian_kernal /= np.max(long_edge_gaussian_kernal)
            short_edge_gaussian_kernal = cv2.getGaussianKernel(norm_static.shape[1],norm_static.shape[1]/2)
            short_edge_gaussian_kernal /= np.max(short_edge_gaussian_kernal)
            norm_static = long_edge_gaussian_kernal * norm_static * short_edge_gaussian_kernal.T
        S_bar = cv2.resize(norm_static,(width,height),interpolation=cv2.INTER_NEAREST)
        image_saliency_map[:,:,_count]=np.float16(S_bar)
        
    video_cap.release()
    cv2.destroyAllWindows() 
    video_index = video_path.split("\\")[-1].split(".")[0]
    with h5py.File("h5rst\\adjusted_image_saliency_map.h5", "a") as hf:
        hf.create_dataset(f"{video_index}_image_saliency_map", data=image_saliency_map, compression='gzip', compression_opts=9)
        print(len(list(hf.keys())))

if __name__ == "__main__":
    args = parse_args()
    main(args)
    