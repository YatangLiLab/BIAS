import cv2
import argparse
import tqdm
import time
import numpy as np
import matplotlib.pyplot as plt
from utils import read_video, normalize_img, addition,interference_function, initialize_gabor_lib,load_maximum_dll,multation
from newStaticSaliency import resize_to_normal_shape, Itti_down_sampling, Itti_feature_maps, synthesis_conspicuous_map
from selfMovement import four_dir_sim, Intensity_processing, Camera
from flicker import flicker_computation
from orientation_saliency import O_bar_synthesis
import os

def parse_args():
    """
    get args.
    """
    parse = argparse.ArgumentParser(description='essential parameters') 
    # 
    parse.add_argument('--video_path', default="E:\\Li Lab\\itti_and_lif\\video\\video\\video\\230.avi", type=str, help='path of sample video') 
    # parse.add_argument('--video_path', default="E:\\Li Lab\\itti_and_lif\\CarCrashVideo\\sample_video.mp4", type=str, help='path of sample video') 
    parse.add_argument('--generate_name',default = "result_video\\230_newnew.mp4",type=str,help = 'default generate name')
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
    parse.add_argument('--alpha_beta',default = (0.3, 0.3),type=lambda s:tuple(float(item) for item in s.split(',')),help = 'determine standard alpha and beta.')
    parse.add_argument('--output',default="quad",type = str, help = "determine output format.")
    parse.add_argument('--target_folder',default='optimize',type=str,help = 'default place to place pred results')
    parse.add_argument('--center',default = (1,),type=lambda s:tuple(int(item) for item in s.split(',')[:-1]),help = "center params. Itti default params are (2, 3, 4), equals to (1,2,3) here, in fast paras they use (0,)")
    parse.add_argument('--surrounding',default = (4,),type=lambda s:tuple(int(item) for item in s.split(',')[:-1]),help = "surrounding params. Itti default params are (3, 4),in fast pami they use (4,)")
    parse.add_argument('--motion_center',default = (1,2,3),type=lambda s:tuple(int(item) for item in s.split(',')),help = "center params. Itti default params are (2, 3, 4)(1,2,3)")
    parse.add_argument('--motion_surrounding',default = (3,4),type=lambda s:tuple(int(item) for item in s.split(',')),help = "surrounding params. Itti default params are (3, 4)(3,4)")
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
    #print(args)
    #quit()
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
    fourcc = cv2.VideoWriter.fourcc(*"MP4V")
    fps = video_cap.get(cv2.CAP_PROP_FPS)
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
    control_list = [Is]*args.control_length

    if args.output == "write" or args.output == "test":
        writer = cv2.VideoWriter(args.generate_name,fourcc,fps,final_shape) # decided by args.double_height
    elif args.output == "save_img":    
        f_time1 = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        tag = video_path.split("\\")[-1].split("/")[-1].split(".")[0]
        folder_name = f"{args.target_folder}\\{tag}"
        os.mkdir(folder_name)
    elif args.output == 'quad':
        final_shape = (width*2,height*2)
        writer = cv2.VideoWriter(args.generate_name,fourcc,fps,final_shape)
    else:
        raise NotImplementedError(f"args.output should in ['write', 'save_img'], not {args.output}")

    # calculate first image only based on static methods.
    I_dict, RG_dict, BY_dict = Itti_feature_maps(rIs, Rs, Gs, Bs, Ys, rDs, norm_lib, args)
    I_bar, C_bar = synthesis_conspicuous_map(I_dict, RG_dict, BY_dict,norm_lib,args)
    I_bar, C_bar = normalize_img(I_bar,norm_lib),normalize_img(C_bar,norm_lib)
    O_bar = normalize_img(O_bar_synthesis(Is.copy(),args,gabor_lib,norm_lib),norm_lib)
    static_saliency_map = addition(I_bar/(np.sum(I_bar)+1e-3)+ C_bar/(np.sum(C_bar)+1e-3), (O_bar/np.sum(O_bar)+1e-3))
    static_saliency_map = cv2.resize(static_saliency_map, (width,height))
    static_saliency_map = cv2.normalize(static_saliency_map, None, 0, 255, cv2.NORM_MINMAX) # minmax coding

    rgb = cv2.cvtColor(static_saliency_map,cv2.COLOR_GRAY2BGR)
    fig,axs = plt.subplots(2,2)
    if args.double_height:
        # show complex image
        out_image = np.concatenate([first_frame, rgb],0)
        if args.output == "write":
            writer.write(out_image)
        elif args.output == 'save_img':
            cv2.imwrite(f"{folder_name}\\{1:04d}.png",out_image)
        elif args.output == "test":
            writer.write(out_image)
        else:
            raise NotImplementedError
    else:
        if args.output == "write":
            writer.write(np.uint8(rgb))
        elif args.output == 'save_img':          
            cv2.imwrite(f"{folder_name}\\{1:04d}.png",rgb)
        elif args.output == "test":
            writer.write(np.uint8(rgb//4 * 3 + first_frame // 4))
        elif args.output == 'quad':
            gt = cv2.imread(f"e:\\Li Lab\\itti_and_lif\\video\\annotation\\0{video_path[-7:-4]}\\maps\\0001.png")
            out_image_left = np.concatenate([rgb//4 * 3 + first_frame // 4, rgb//4 * 3 + first_frame // 4],0) # our result and itti's result
            out_image_right = np.concatenate([gt//4 * 3 + first_frame // 4,first_frame],0)
            out_image = np.uint8(np.concatenate([out_image_left, out_image_right],1))
            writer.write(out_image)
        else:
            raise NotImplementedError
    #------------------------------------------------VideoProcessing---------------------------------------------------------
    time_list = [[],[],[],[],[]] # analyze amounts of time for generate gabor_Pyramid, eight_dir_processing and addition.
    camera_motion_lst = [Camera()] * len(args.default_checkpoint) # initialize Camera

    for _count in tqdm.trange(int(frame_number)): # in range(int(frame_number)): #
        ret, frame = video_cap.read()
        if not ret: # whole video is processed
            print("Done!")
            break
        time0 = time.time()
        resized_image = cv2.resize(frame,(320,240),interpolation=cv2.INTER_NEAREST)# resize_to_normal_shape(frame)

        rIs, Rs, Gs, Bs, Ys, rDs, Is = Itti_down_sampling(resized_image,args) # intensity channels and color channels
        time01 = time.time() 
        
        # DownSampleing Pyramids
        
        I_dict, RG_dict, BY_dict = Itti_feature_maps(rIs, Rs, Gs, Bs, Ys, rDs,norm_lib ,args)
        I_bar, C_bar = synthesis_conspicuous_map(I_dict, RG_dict, BY_dict,norm_lib,args)
        I_bar, C_bar = synthesis_conspicuous_map(I_dict, RG_dict, BY_dict,norm_lib,args)
        time05 = time.time()
        #generate Intensity and color Saliency map


        O_bar = normalize_img(O_bar_synthesis(Is.copy(),args,gabor_lib,norm_lib),norm_lib)
        static_saliency_map = addition(I_bar/(np.sum(I_bar)+1e-3)+ C_bar/(np.sum(C_bar)+1e-3), (O_bar/np.sum(O_bar)+1e-3))
        # now we have finished calculating static saliency map.

        control_list.pop()
        # insert new pyramid
        control_list.insert(0,Is)
        time1 = time.time()
        # generate orientation saliency map
        maps = []
        flicker_maps = []
        for index, checkpoint in enumerate(args.default_checkpoint):
        #    # print(len(control_list[0]))
            results = four_dir_sim(control_list[0],control_list[checkpoint-1],args,camera_motion_lst[index])
            motion_saliency_map = np.zeros((1,1))
            for single_map in results:
                motion_saliency_map = addition(motion_saliency_map,normalize_img(Intensity_processing(single_map,norm_lib,args),norm_lib))
            maps.append(motion_saliency_map * args.decay_factor ** index)

            flicker_rsts = flicker_computation(control_list[0],control_list[checkpoint-1],args)
            tmp_flicker_map = np.zeros_like(flicker_rsts[0])
            for single_map in flicker_rsts:
                tmp_flicker_map = addition(tmp_flicker_map,normalize_img(single_map,norm_lib))
            flicker_maps.append(tmp_flicker_map)
            # diff_time_scale_map = addition(diff_time_scale_map,normalize_img(eight_dir_processing(control_list[0],control_list[checkpoint])))
        diff_time_scale_map = sum(maps) if len(maps) > 1 else maps[0]
        flicker_final_map = sum(flicker_maps) if len(flicker_maps) > 1 else flicker_maps[0]
        ###  Modified --- Area ###
        
        time2 = time.time()
            # diff_time_scale_list.append() # add specific checkpoint results
        # diff_time_scale_map = cv2.resize(diff_time_scale_map,(width,height))

        # plt.imshow(diff_time_scale_map)
        # plt.show()
        # fig,((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2)
        # ax1.imshow(normalize_img(I_bar,norm_lib))
        # ax1.set_title("Ibar")
        # ax2.imshow(normalize_img(C_bar,norm_lib))
        # ax2.set_title("Cbar")
        # ax3.imshow(normalize_img(O_bar,norm_lib))
        # ax3.set_title("Obar")
        # ax4.imshow(normalize_img(diff_time_scale_map,norm_lib))
        # ax4.set_title("diff_time_scale_map")
        # plt.show()
        norm_static, norm_dynamic = normalize_img(static_saliency_map,norm_lib), normalize_img(diff_time_scale_map,norm_lib)
        norm_flicker = normalize_img(flicker_final_map,norm_lib)
        if args.add_central_gaussian:
            long_edge_gaussian_kernal = cv2.getGaussianKernel(norm_dynamic.shape[0],norm_dynamic.shape[0]/2)
            long_edge_gaussian_kernal /= np.max(long_edge_gaussian_kernal)
            short_edge_gaussian_kernal = cv2.getGaussianKernel(norm_dynamic.shape[1],norm_dynamic.shape[1]/2)
            short_edge_gaussian_kernal /= np.max(short_edge_gaussian_kernal)

            norm_dynamic = long_edge_gaussian_kernal * norm_dynamic * short_edge_gaussian_kernal.T

            long_edge_gaussian_kernal = cv2.getGaussianKernel(norm_static.shape[0],norm_static.shape[0]/2)
            long_edge_gaussian_kernal /= np.max(long_edge_gaussian_kernal)
            short_edge_gaussian_kernal = cv2.getGaussianKernel(norm_static.shape[1],norm_static.shape[1]/2)
            short_edge_gaussian_kernal /= np.max(short_edge_gaussian_kernal)
            norm_static = long_edge_gaussian_kernal * norm_static * short_edge_gaussian_kernal.T

            long_edge_gaussian_kernal = cv2.getGaussianKernel(norm_flicker.shape[0],norm_flicker.shape[0]/2)
            long_edge_gaussian_kernal /= np.max(long_edge_gaussian_kernal)
            short_edge_gaussian_kernal = cv2.getGaussianKernel(norm_flicker.shape[1],norm_flicker.shape[1]/2)
            short_edge_gaussian_kernal /= np.max(short_edge_gaussian_kernal)
            norm_flicker = long_edge_gaussian_kernal * norm_flicker * short_edge_gaussian_kernal.T

            
        #norm_static = np.where(norm_static > np.percentile(norm_static,args.selective_threshold),norm_static,0)
        #norm_dynamic = np.where(norm_dynamic > np.percentile(norm_dynamic,args.selective_threshold),norm_dynamic,0)
        if args.continuity:
            if _count == 0:
                tmp_static = norm_static / max((1-tmp_save_factor),1e-6)
                tmp_dynamic = norm_dynamic / max((1-tmp_save_factor),1e-6)
                tmp_flicker = norm_flicker / max((1-tmp_save_factor),1e-6)
            else:
                tmp_static = tmp_static * tmp_save_factor + norm_static
                tmp_dynamic = tmp_dynamic * tmp_save_factor + norm_dynamic
                tmp_flicker = tmp_flicker * tmp_save_factor + norm_flicker
                norm_static = tmp_static * (1-tmp_save_factor)
                norm_dynamic = tmp_dynamic * (1-tmp_save_factor)
                norm_flicker = tmp_flicker * (1-tmp_save_factor)
        # S_bar = multation((alpha + 10 * norm_static),(beta + 10 * norm_dynamic))
        if args.generate_type == 'static':
            norm_dynamic *= 0
        elif args.generate_type == 'dynamic':
            norm_static *= 0
        elif args.generate_type == 'both':
            pass
        else:
            raise RuntimeError(f"not defined generatetype of {args.generate_type}")
        S_bar = multation(alpha + norm_static , beta + norm_dynamic) - alpha * beta
        S_bar = addition(S_bar, norm_flicker)
        # S_bar = np.where(S_bar > np.percentile(S_bar,args.selective_threshold),S_bar,0)
        # S_bar = addition(norm_static,norm_dynamic)
        #a_sd = 2.80
        #a_s = 0.69
        #a_d = 0.84
        #a_constant = 0
        #S_bar = addition(a_s * norm_static,addition(a_d * norm_dynamic,\
        #                a_sd*multation(norm_dynamic,norm_static))) + a_constant
        #S_bar = np.uint8(np.clip(S_bar,0,255))
        
        S_bar = np.uint8((S_bar - np.min(S_bar))/(np.max(S_bar) - np.min(S_bar)+1e-4) * 255)
        # S_bar = cv2.normalize(S_bar, None, 0, 255, cv2.NORM_MINMAX)
        S_bar = cv2.resize(S_bar,(width,height),interpolation=cv2.INTER_NEAREST)
        #S_bar = interference_function(S_bar, (height//2,width//2)) # add central gaussian 
        S_bar = cv2.normalize(S_bar,None, 0, 255, cv2.NORM_MINMAX) if np.max(S_bar)>0 else S_bar * 0
        S_bar = cv2.GaussianBlur(S_bar,(15,15),sigmaX=0.04*640,sigmaY=0.04*640)
        S_bar = np.where(S_bar > np.percentile(S_bar,args.selective_threshold),S_bar,0)

        # fig,((ax1,ax2,ax3),(ax4,ax5,ax6)) = plt.subplots(2,3)
        # ax1.imshow(normalize_img(I_bar,norm_lib))
        # ax1.set_title("Ibar")
        # ax2.imshow(normalize_img(C_bar,norm_lib))
        # ax2.set_title("Cbar")
        # ax4.imshow(normalize_img(O_bar,norm_lib))
        # ax4.set_title("Obar")
        # ax3.imshow(norm_dynamic)#normalize_img(diff_time_scale_map,norm_lib))
        # ax3.set_title("diff_time_scale_map")
        # ax5.imshow(norm_static)#static_saliency_map)#normalize_img(static_saliency_map,norm_lib))
        # ax5.set_title("Static Saliency map")
        # ax6.imshow(S_bar)
        # ax6.set_title("S_bar")
        # plt.show()
        

        if args.output_gamma_correlation:
            gamma = args.gamma_correlation
            invGamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
	            # apply gamma correction using the lookup table
            S_bar =  cv2.LUT(S_bar, table)
            S_bar = cv2.GaussianBlur(S_bar,(5,5),sigmaX=0.02*S_bar.shape[1],sigmaY=0.02*S_bar.shape[0])

        rgb = cv2.cvtColor(np.uint8(S_bar),cv2.COLOR_GRAY2BGR)

        if args.double_height:
        # show complex image
            out_image = np.concatenate([frame, rgb],0)
            if args.output == "write":
                writer.write(out_image)
            elif args.output == 'save_img':
                cv2.imwrite(f"{folder_name}\\{_count+2:04d}.png",out_image)
            else:
                raise NotImplementedError
        else:
            if args.output == "write":
                # writer.write(rgb//4 * 3 + frame // 4)
                writer.write(rgb)
            elif args.output == 'save_img':
                cv2.imwrite(f"{folder_name}\\{_count+2:04d}.png",rgb)
            elif args.output == "test":
                writer.write(rgb//4 * 3 + frame // 4)
            elif args.output == 'quad':
                gt = cv2.imread(f"e:\\Li Lab\\itti_and_lif\\video\\annotation\\0{video_path[-7:-4]}\\maps\\{_count+2:04d}.png")
                itti_saliency = normalize_img(static_saliency_map,norm_lib)
                itti_saliency = (itti_saliency - np.min(itti_saliency))/(np.max(itti_saliency) - np.min(itti_saliency) + 1e-6) * 255
                itti_saliency = cv2.resize(itti_saliency,(width,height),interpolation=cv2.INTER_NEAREST)
                #S_bar = interference_function(S_bar, (height//2,width//2)) # add central gaussian 
                itti_saliency = cv2.normalize(itti_saliency,None, 0, 255, cv2.NORM_MINMAX) if np.max(itti_saliency)>0 else itti_saliency * 0
                itti_saliency = cv2.GaussianBlur(itti_saliency,(15,15),sigmaX=0.04*640,sigmaY=0.04*640)
                itti_saliency = np.where(itti_saliency > np.percentile(itti_saliency,args.selective_threshold),itti_saliency,0)
                itti_saliency = cv2.cvtColor(np.uint8(itti_saliency),cv2.COLOR_GRAY2BGR)

                motion_saliency = (norm_dynamic - np.min(norm_dynamic))/(np.max(norm_dynamic)-np.min(norm_dynamic)+1e-6)*255
                motion_saliency = cv2.resize(motion_saliency,(width,height),interpolation=cv2.INTER_NEAREST)
                #S_bar = interference_function(S_bar, (height//2,width//2)) # add central gaussian 
                motion_saliency = cv2.normalize(motion_saliency,None, 0, 255, cv2.NORM_MINMAX) if np.max(motion_saliency)>0 else itti_saliency * 0
                motion_saliency = cv2.GaussianBlur(motion_saliency,(15,15),sigmaX=0.04*640,sigmaY=0.04*640)
                motion_saliency = np.where(motion_saliency > np.percentile(motion_saliency,args.selective_threshold),motion_saliency,0)
                motion_saliency = cv2.cvtColor(np.uint8(motion_saliency),cv2.COLOR_GRAY2BGR)
                out_image_left = np.concatenate([motion_saliency//4 * 3 + frame // 4, itti_saliency//4 * 3 + frame // 4],0) # our result and itti's result
                out_image_right = np.concatenate([gt//4 * 3 + frame // 4,rgb//4 * 3 + frame // 4],0)
                out_image = np.uint8(np.concatenate([out_image_left, out_image_right],1))
                writer.write(out_image)
            else:
                raise NotImplementedError
        time3 = time.time()

#         fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2)
#         ax1.imshow(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
#         ax1.set_title("Original frame")
#         ax2.imshow(np.float32(rgb//4 * 3 + cv2.cvtColor(frame,cv2.COLOR_BGR2RGB) // 4)/255)
#         ax2.set_title("Pred + frame")
#         ax3.imshow(norm_static)
#         ax3.set_title("static pred")
#         ax4.imshow(norm_dynamic)
#         ax4.set_title("dynamic pred")
#         plt.show()
#         time_list[0].append(time01-time0)
#         time_list[1].append(time05-time01)
#         time_list[2].append(time1-time05)
#         time_list[3].append(time2-time1)
#         time_list[4].append(time3-time2)
#         print(\
# f"\n-----------------------------------------------------------------------------------------\n\
# here are the timing results:\n\
# for down_sampling_Pyramid: {np.mean(time_list[0])} seconds\n\
# for generate Intensity & color saliency map: {np.mean(time_list[1])} seconds\n\
# for generate orientation saliency map: {np.mean(time_list[2])} seconds\n\
# for calculate motion saliency: {np.mean(time_list[3])} seconds\n\
# and addition and other stuffs: {np.mean(time_list[4])} seconds.\n\
# -----------------------------------------------------------------------------------------\n")
        
    video_cap.release()
    cv2.destroyAllWindows() 


if __name__ == "__main__":
    args = parse_args()
    main(args)
    