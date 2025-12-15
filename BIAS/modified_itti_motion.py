import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
from tqdm import trange
from utils import *
from .image_saliency import Is_scale
import time

def zero_edge(image):
    image[:1, :] = image[-1:, :] = image[:, :1] = image[:, -1:] = 0
    return image

def timer(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Function {func.__name__} took {end_time - start_time} seconds to run.")
        return result
    return wrapper

def resize_to_normal_shape(image):
    """
    Input is provided in the form of static color images, usually digitized at 640*480 resolution
    """
    resized_image = cv2.resize(image,(640,480),interpolation=cv2.INTER_NEAREST)
    return resized_image

def seperate_RGB_chanells(image):
   """
    Seperate RGB chanells to going through following algorithms
    """
   return image[:,:,0], image[:,:,1], image[:,:,2]

def Intensity_processing(image,norm_lib,args,ifshow = False):
    Is = eight_pyrimid_built(image)
    Ds = [np.max(single_I) - single_I for single_I in Is]

    #Is = [np.where(I - np.mean(I) > 0,I - np.mean(I),0) for I in Is]
    #Ds = [np.where(D - np.mean(D) > 0,D - np.mean(D),0) for D in Is]

    c_set = (2,)#args.center
    delta_set = (3,)#args.surrounding
    I_dict = {}
    for c in c_set:
        for delta in delta_set:
            I_dict[(c,c+delta)] = Is_scale(Is,Ds,c,c+delta)
    I_bar = np.zeros((1,1),dtype=np.float32)
    for c in c_set:
        for delta in delta_set:
            assert I_dict[(c,c+delta)][0].shape == I_dict[(c,c+delta)][1].shape
            I_bar = addition(I_bar,normalize_img(np.float32(I_dict[(c,c+delta)][0]),norm_lib))  # White
            I_bar = addition(I_bar,normalize_img(np.float32(I_dict[(c,c+delta)][1]),norm_lib))  # Black
            # darkness term# 
    if ifshow:
        # plt.imshow(I_bar)
        fig, (ax1,ax2) = plt.subplots(1,2)
        ax1.imshow(image)
        ax2.imshow(normalize_img(I_bar,norm_lib))
        plt.show()
    return I_bar
  
# def dir_8_motion(Is0,Is1,args,ifshow = False):
#     """
#     we want to make a simple biological plausible motion detection method.
#     """
#     # motion_map = frame_1 - frame_0
#     rs, ls, us, ds = [], [], [], []
#     urs, uls, drs, dls = [], [], [], []
#     for i in (2,4):
#             max_form = np.where(Is1[i]>Is0[i],Is1[i],Is0[i])
#             shape = Is1[i].shape
#             tmp = np.abs(Is1[i] - Is0[i])
#             # tmp = np.where(tmp != 0, 1, 0)
#             # tmp = np.where(tmp>np.percentile(tmp,50),1,0)
#             #diff_against_background = np.abs(Is1[i] - np.mean(Is1[i]))
#             sim_map_i = np.exp(-np.abs(tmp))
#             # horizontal differentiation:
#             # dx_image_r = np.ones_like(Is1[i]) * 100
#             dx_image_r[:,1:shape[1]] = Is1[i][:,1:shape[1]] - Is0[i][:,0:shape[1]-1]
#             dx_image_r /= max_form + 1e-6
#             #dy_image_d *= diff_against_background
#             # dx_image_l = np.ones_like(Is1[i]) * 100
#             dx_image_l[:,0:shape[1]-1] = Is1[i][:,0:shape[1]-1] - Is0[i][:,1:shape[1]]
#             dx_image_l /= max_form + 1e-6
#             #dy_image_u *= diff_against_background
#             # dy_image_d = np.ones_like(Is1[i]) * 100
#             dy_image_d[1:shape[0],:] = Is1[i][1:shape[0],:] - Is0[i][0:shape[0]-1,:]
#             dy_image_d /= max_form + 1e-6
#             #dx_image_r *= diff_against_background
#             # dy_image_u = np.ones_like(Is1[i]) * 100
#             dy_image_u[0:shape[0]-1,:] = Is1[i][0:shape[0]-1,:] - Is0[i][1:shape[0],:]
#             dy_image_u /= max_form + 1e-6
#             #dx_image_l *= diff_against_background
#             factor = 0.5

#             # dx_image_ur = np.ones_like(Is1[i]) * 100
#             dx_image_ur[:,1:shape[1]] = Is1[i][0:shape[0]-1,1:shape[1]] - Is0[i][1:shape[0],0:shape[1]-1]
#             dx_image_ur /= max_form + 1e-6
#             #dy_image_d *= diff_against_background
#             # dx_image_ul = np.ones_like(Is1[i]) * 100
#             dx_image_ul[:,0:shape[1]-1] = Is1[i][0:shape[0]-1,0:shape[1]-1] - Is0[i][1:shape[0],1:shape[1]]
#             dx_image_ul /= max_form + 1e-6
#             #dy_image_u *= diff_against_background
#             # dy_image_dr = np.ones_like(Is1[i]) * 100
#             dy_image_dr[1:shape[0],:] = Is1[i][1:shape[0],1:shape[1]] - Is0[i][0:shape[0]-1,0:shape[1]-1]
#             dy_image_dr /= max_form + 1e-6
#             #dx_image_r *= diff_against_background
#             # dy_image_dl = np.ones_like(Is1[i]) * 100
#             dy_image_dl[0:shape[0]-1,:] = Is1[i][1:shape[0],0:shape[1]-1] - Is0[i][0:shape[0]-1,1:shape[1]]
#             dy_image_dl /= max_form + 1e-6
#             #dx_image_l *= diff_against_background
#             factor = 0.5
#             # norm = Normalize(vmin = 0, vmax = max(np.max(np.exp(-np.abs(dx_image_r)/factor)*tmp),np.max(np.exp(-np.abs(dx_image_l)/factor)*tmp)\
#             #                                        ,np.max(np.exp(-np.abs(dy_image_u)/factor)*tmp),np.max(np.exp(-np.abs(dy_image_d)/factor)*tmp)))
#             # fig, ((ax1,ax2,ax5),(ax3,ax4,ax6)) = plt.subplots(2,3)
            
#             # ax1.imshow(np.exp(-np.abs(dx_image_r)/factor)*tmp, cmap='viridis')
#             # ax1.set_title("r")
#             # ax2.imshow(np.exp(-np.abs(dx_image_l)/factor)*tmp, cmap='viridis')
#             # ax2.set_title("l")
#             # ax3.imshow(np.exp(-np.abs(dy_image_u)/factor)*tmp, cmap='viridis')
#             # ax3.set_title("u")
#             # ax4.imshow(np.exp(-np.abs(dy_image_d)/factor)*tmp, cmap='viridis')
#             # #ax4.imshow(dy_image_d)
#             # ax4.set_title("d")
#             # ax5.imshow(Is1[i])
#             # ax6.imshow(tmp)
#             # plt.suptitle(f"resolution{i}")
#             # plt.show()
#             r, l, u, d = np.exp(-np.abs(dx_image_r)/50)*tmp, np.exp(-np.abs(dx_image_l)/50)*tmp, np.exp(-np.abs(dy_image_u)/50)*tmp,np.exp(-np.abs(dy_image_d)/50)*tmp
#             ur, ul, dr, dl = np.exp(-np.abs(dx_image_ur)/50)*tmp, np.exp(-np.abs(dx_image_ul)/50)*tmp, np.exp(-np.abs(dy_image_dr)/50)*tmp,np.exp(-np.abs(dy_image_dl)/50)*tmp
#             rs.append(r)
#             ls.append(l)
#             us.append(u)
#             ds.append(d)
#             urs.append(ur)
#             uls.append(ul)
#             drs.append(dr)
#             dls.append(dl)
            
#     tu, td = Is_scale(us,ds,0,1)
#     tl, tr = Is_scale(ls,rs,0,1)
#     tur, tdl = Is_scale(urs,dls,0,1)
#     tul, tdr = Is_scale(uls,drs,0,1)
#     if ifshow:
#         fig, ((ax1,ax2,ax5),(ax3,ax4,ax6)) = plt.subplots(2,3)
#         # norm = Normalize(vmin=0, vmax=(max(list(map(np.max,[tr,td,tl,tr])))))
#         ax1.imshow(tr, cmap='viridis')
#         ax1.set_title("r")
#         ax2.imshow(tl, cmap='viridis')
#         ax2.set_title("l")
#         ax3.imshow(tu, cmap='viridis')
#         ax3.set_title("u")
#         ax4.imshow(td, cmap='viridis')
#         # ax4.imshow(dy_image_d)
#         ax4.set_title("d")
#         ax5.imshow(Is1[i])
#         ax5.set_title("source image")
#         ax6.imshow(output := sum(list(map(normalize_img, [tr,tl,tu,td]))))
#         ax6.set_title("out image")
#         plt.show()
#     return tr, tl, tu, td, tur, tdl, tul, tdr

def eight_dir_sim(Is0,Is1,args,ifshow = False   )->np.array:
    """
    use Hassenstein-Reichardt-like model to detect motion information.
    """
    

    directions = [(-1,-1),(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1)] # maybe use args one day......
    
    # pad the former frame, make it easier for future calculations
    motion_dict = {}
    motion_dict["rs"] = []
    motion_dict["ls"] = []
    motion_dict["us"] = []
    motion_dict["ds"] = []
    motion_dict["urs"] = []
    motion_dict["uls"] = []
    motion_dict["drs"] = []
    motion_dict["dls"] = []
    motion_list = ["drs","ds","dls","ls","uls","us","urs","rs"]
    results = []
    half = len(directions)//2
    # static_map = np.ones_like((1,1))
    for spatial_index in (1,3):
        current_frame = np.float32(Is0[spatial_index])
        former_frame = np.float32(Is1[spatial_index])
        max_form = cv2.GaussianBlur(np.where(current_frame>former_frame,current_frame,former_frame),(5,5),sigmaX= 0.04 * current_frame.shape[0])
        tmp = cv2.GaussianBlur(np.abs(current_frame - former_frame),(3,3),sigmaX= 0.02 * current_frame.shape[0]) / (max_form+1e-3)
        #tmp = np.where(tmp>np.percentile(tmp,75),1,0)
        # plt.imshow(tmp)
        # plt.show()

        
        shape = current_frame.shape
        padded_former_frame = np.pad(former_frame,1,mode='reflect')
        for dir_pair_index in range(half):
            l_pair = np.abs(current_frame - padded_former_frame[directions[dir_pair_index][0]+1 : directions[dir_pair_index][0]+1+shape[0],\
                                                                directions[dir_pair_index][1]+1 : directions[dir_pair_index][1]+1+shape[1]])
            r_pair = np.abs(current_frame - padded_former_frame[directions[half + dir_pair_index][0]+1 : directions[half + dir_pair_index][0]+1+shape[0],\
                                                                directions[half + dir_pair_index][1]+1 : directions[half + dir_pair_index][1]+1+shape[1]])
            padd_r_pair = np.pad(r_pair,1,mode='reflect')
            l, r = np.exp(-l_pair/50), np.exp(-1/50 * padd_r_pair[directions[dir_pair_index][0]+1 : directions[dir_pair_index][0]+1+shape[0],\
                                                                                      directions[dir_pair_index][1]+1 : directions[dir_pair_index][1]+1+shape[1]])
            motion_dict[motion_list[dir_pair_index]].append(zero_edge(l * tmp))
            motion_dict[motion_list[half + dir_pair_index]].append(zero_edge(r * tmp))
            #fig,((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2)
            #ax1.imshow(l_pair)
            #ax1.set_title(f"current - {motion_list[dir_pair_index]}")
            #ax2.imshow(r_pair)
            #ax2.set_title(f"current - {motion_list[dir_pair_index+half]}")
            #ax3.imshow(l)
            #ax4.imshow(r)
            #plt.show()
            # static_map = sub_img2(static_map,l_pair+ r_pair)
            # plt.imshow(static_map)
            # plt.show()
    # static_map = cv2.normalize(static_map,None,0,1,cv2.NORM_MINMAX)

    for dir_pair_index in range(half):
            tl,tr = Is_scale(motion_dict[motion_list[dir_pair_index]],motion_dict[motion_list[half + dir_pair_index]],0,1)
            results.append(tl)#cv2.GaussianBlur(tl,(5,5),3))
            results.append(tr)#cv2.GaussianBlur(tr,(5,5),3)) 
            #if ifshow:
            #    fig,axs = plt.subplots(4,2)
            #    axs[0,0].imshow(motion_dict[motion_list[dir_pair_index]][0])
            #    axs[0,0].set_title(motion_list[dir_pair_index])
            #    axs[0,1].imshow(motion_dict[motion_list[half + dir_pair_index]][0])
            #    axs[0,1].set_title(motion_list[dir_pair_index+half])
            #    axs[1,0].imshow(motion_dict[motion_list[dir_pair_index]][1])
            #    axs[1,1].imshow(motion_dict[motion_list[half + dir_pair_index]][1])
            #    axs[2,0].imshow(np.abs(tl))
            #    axs[2,1].imshow(np.abs(tr))
            #    axs[3,0].imshow(-np.log(motion_dict[motion_list[dir_pair_index]][0]+1e-3))
            #    axs[3,1].imshow(-np.log(motion_dict[motion_list[half + dir_pair_index]][0]+1e-3))
            #    plt.show()

            # results.append(np.abs(tl - tr))
    
    # results.append(static_map)
    if ifshow:
        fig,axs = plt.subplots(3,5)
        axs[0,0].imshow(l_pair)
        axs[1,0].imshow(r_pair)
        axs[2,0].imshow(l_pair+r_pair)
        for col in range(1,1 + half):
            axs[0,col].imshow(results[2*col-2] + results[2*col -1])
            axs[0,col].set_title(f"{directions[col-1]} + {directions[col + half - 1]}")
            axs[1,col].imshow(results[2*col-2])
            axs[1,col].set_title(f"{directions[col-1]}")
            axs[2,col].imshow(results[2*col -1])
            axs[2,col].set_title(f"{directions[col + half - 1]}")
        plt.show()
    return results


def four_dir_sim(Is0,Is1,args,ifshow = False)->np.array:
    """
    use Hassenstein-Reichardt-like model to detect motion information.
    """

    directions = [(-1,0),(0,1),(1,0),(0,-1)] # maybe use args one day......
    
    # pad the former frame, make it easier for future calculations
    motion_dict = {}
    motion_dict["rs"] = []
    motion_dict["ls"] = []
    motion_dict["us"] = []
    motion_dict["ds"] = []
    motion_dict['stable'] = []
    motion_list = ["ds","ls","us","rs",'stable']
    results = []
    half = len(directions)//2
    # static_map = np.ones_like((1,1))
    for spatial_index in (1,3):
        current_frame = np.float32(Is0[spatial_index])
        former_frame = np.float32(Is1[spatial_index])
        max_form = np.where(current_frame>former_frame,current_frame,former_frame)
        tmp = cv2.GaussianBlur(np.abs(current_frame - former_frame),(3,3),sigmaX= 0.02 * current_frame.shape[0]) / (max_form+1e-3)
        #tmp = np.where(tmp>np.percentile(tmp,75),1,0)
        # plt.imshow(tmp)
        # plt.show()

        stable_img = np.exp(-np.abs(current_frame/(max_form+25) - former_frame/(max_form+25))*10) * np.mean(current_frame)
        if ifshow:
            fig,((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2)
            ax1.imshow(current_frame/(max_form+25))
            ax2.imshow(former_frame/(max_form+25))
            ax3.imshow(-np.abs(current_frame/(max_form+25) - former_frame/(max_form+25)))
            ax4.imshow(stable_img)
            plt.show()
        #plt.imshow(stable_img)
        #plt.show()
        shape = current_frame.shape
        padded_former_frame = np.pad(former_frame,1,mode='reflect')
        for dir_pair_index in range(half):
            l_pair = np.abs(current_frame - padded_former_frame[directions[dir_pair_index][0]+1 : directions[dir_pair_index][0]+1+shape[0],\
                                                                directions[dir_pair_index][1]+1 : directions[dir_pair_index][1]+1+shape[1]])
            r_pair = np.abs(current_frame - padded_former_frame[directions[half + dir_pair_index][0]+1 : directions[half + dir_pair_index][0]+1+shape[0],\
                                                                directions[half + dir_pair_index][1]+1 : directions[half + dir_pair_index][1]+1+shape[1]])
            padd_r_pair = np.pad(r_pair,1,mode='reflect')
            l, r = np.exp(-l_pair/50) * tmp, np.exp(-1/50 * padd_r_pair[directions[dir_pair_index][0]+1 : directions[dir_pair_index][0]+1+shape[0],\
                                                                                      directions[dir_pair_index][1]+1 : directions[dir_pair_index][1]+1+shape[1]])*tmp
            l, r = cv2.GaussianBlur(l,(3,3),2), cv2.GaussianBlur(r,(3,3),2)
            l, r = np.where(l>np.percentile(l,90),l,0),np.where(r>np.percentile(r,90),r,0)
            stable_img -= np.abs(l-r) * np.mean(current_frame)
            motion_dict[motion_list[dir_pair_index]].append(zero_edge(l))
            motion_dict[motion_list[half + dir_pair_index]].append(zero_edge(r))
        stable_img = np.where(stable_img > 0, stable_img,0)
        motion_dict['stable'].append(stable_img)

    tu, td = Is_scale(motion_dict['us'],motion_dict['ds'],0,1)
    tl, tr = Is_scale(motion_dict['ls'],motion_dict['rs'],0,1)
    stable, not_stable = Is_scale(motion_dict['stable'],[255 - motion_dict['stable'][index] for index in range(len(motion_dict['stable']))],0,1)
    if ifshow:
        norm = Normalize(vmin = 0, vmax = max(np.max(tu),np.max(td),np.max(tl),np.max(tr)))
        fig, ((ax1,ax2,ax5),(ax3,ax4,ax6)) = plt.subplots(2,3)
        ax1.imshow(tr, cmap='viridis', norm=norm)
        ax1.set_title("tr")
        ax2.imshow(tl, cmap='viridis', norm=norm)
        ax2.set_title("tl")
        ax3.imshow(tu, cmap='viridis', norm=norm)
        ax3.set_title("tu")
        ax4.imshow(td, cmap='viridis', norm=norm)
        # ax4.imshow(dy_image_d)
        ax4.set_title("td")
        ax5.imshow(stable)
        ax5.set_title("stable img")
        ax6.imshow(not_stable)
        plt.show()
    results = [tu,td,tl,tr, stable, not_stable]
    return results


def parse_args():
    """
    get args.
    """
    parse = argparse.ArgumentParser(description='essential parameters') 
    parse.add_argument('--video_path', default="E:\\your_path\\RealtimeSaliency\\test_video\\003.avi", type=str, help='path of sample video') 
    parse.add_argument('--generate_name',default = "result_video\\for_test.mp4",type=str,help = 'default generate name')
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
    parse.add_argument('--default_checkpoint', default=(2,),type=tuple,help = 'default checkpoints of control length')
    parse.add_argument('--selective_threshold',default = 0,type=float,help = 'default selective threshold percentage')
    parse.add_argument('--double_height',default = False,type=bool,help = 'default output window')
    parse.add_argument('--alpha_beta',default = (0,1),type=tuple,help = 'determine standard alpha and beta.')
    parse.add_argument('--output',default="write",type = str, help = "determine output format.")
    parse.add_argument('--target_folder',default='lower_result\\tests',type=str,help = 'default place to place pred results')
    parse.add_argument('--center',default = (1,2),type=tuple,help = "center params. Itti default params are (2, 3, 4)")
    parse.add_argument('--surrounding',default = (2,),type=tuple,help = "surrounding params. Itti default params are (3, 4)")
    parse.add_argument('--decay_factor',default = 0.8,type=float,help = "decay factor of time-related problems.")
    parse.add_argument('--gamma_correlation',default=2.2,type = float, help = "gamma value for gamma correlation")
    parse.add_argument('--tmp_save_factor',default=0.9,type = float, help = "gamma value for gamma correlation")
    parse.add_argument('--continuity',type=bool, default=False,help="control how to output, using simple output or continue output.")
    parse.add_argument('--add_central_gaussian',type = bool, default= False, help = "control if apply central gaussian in the end")
    parse.add_argument('--output_gamma_correlation',type = bool, default= False, help = "control if apply central gaussian in the end")
    args = parse.parse_args() 
    return args

if __name__ == "__main__":
    args = parse_args()
    #video_path = "E:\\your_path\\SAVAM\\Videos\\v01_Hugo_2172_left.avi"
    video_path = "E:\\your_path\\RealtimeSaliency\\test_video\\003.avi"
    # video_path = "E:\\your_path\\RealtimeSaliency\\test_video\\test_motion.mp4"
    #video_path = "test_motion.mp4"
    #video_path = "checker.mp4"
    norm_lib = load_maximum_dll()
    video_capture = cv2.VideoCapture(video_path)
    frame_number = video_capture.get(cv2.CAP_PROP_FRAME_COUNT)
    height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    #final_shape = (3*width,2*height)
    final_shape = (width,height)
    fourcc = cv2.VideoWriter.fourcc(*"MP4V")
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    #writer = cv2.VideoWriter(f"check_motion001.mp4",fourcc,fps,final_shape)
    # writer = cv2.VideoWriter(f"notsample009g095-outputgamma-relativemotion.mp4",fourcc,fps,final_shape)
    frame_count = 0
    control_list = []
    gamma = 0.95
    decay_factor = 0.95
    out_frame_list = [] #store gamma-out image
    time_list = (2,4,8,16)
    # time_list = (2,4)
    if video_capture.isOpened():
        for _ in trange(int(300)): # frame_number
            
            ret, frame = video_capture.read()
            if not ret:
                print(f"video finished! total{frame_count}frames.")
                break
            else:
                if frame_count == 0:
                    #control_list.append(np.float32(frame/(1-gamma)))
                        current_frame = resize_to_normal_shape(frame)
                        current_frame = cv2.cvtColor(current_frame,cv2.COLOR_BGR2GRAY)
                        frame_intensity_pyramid = eight_pyrimid_built(current_frame)
                        control_list = [frame_intensity_pyramid] * max(time_list)
                        frame_count += 1
                        continue
                        # control_list.append(np.float32(frame))
                    # frame_count += 1
                    # continue
                current_frame = resize_to_normal_shape(frame)
                current_frame = cv2.cvtColor(current_frame,cv2.COLOR_BGR2GRAY)
                frame_intensity_pyramid = eight_pyrimid_built(current_frame)
                control_list.insert(0,frame_intensity_pyramid)
                control_list.pop(-1)
                output_list = []
                final_output = np.zeros((1,1))
                maps = []
                for index, checkpoint in enumerate(args.default_checkpoint):
        #    # print(len(control_list[0]))
                    results = four_dir_sim(control_list[0],control_list[checkpoint-1],args)
                    motion_saliency_map = np.zeros((1,1))
                    for single_map in results:
                        motion_saliency_map = addition(motion_saliency_map,Intensity_processing(single_map,norm_lib,args))
                        maps.append(motion_saliency_map * args.decay_factor ** index)
            # diff_time_scale_map = addition(diff_time_scale_map,normalize_img(eight_dir_processing(control_list[0],control_list[checkpoint])))
                diff_time_scale_map = sum(maps)
                # diff_time_scale_map = normalize_img(diff_time_scale_map,norm_lib)
                # plt.imshow(diff_time_scale_map)
                # plt.show()
                diff_time_scale_map = diff_time_scale_map/np.max(diff_time_scale_map) * 255 if np.max(diff_time_scale_map) >0 else diff_time_scale_map
                output = cv2.resize(np.uint8(diff_time_scale_map),final_shape,interpolation=cv2.INTER_NEAREST)
                output = cv2.cvtColor(output,cv2.COLOR_GRAY2BGR)
                # plt.imshow(np.float32(output)*3/4/255 + np.float32(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))/4/255)
                # plt.show()
                #for time_ in time_list:
                #    if frame_count != 0:
                #        results = eight_dir_sim(control_list[-1],control_list[-time_],args)
                #        motion_saliency_map = np.zeros((1,1))
                #        for single_map in results:
                #            motion_saliency_map = addition(motion_saliency_map,normalize_img(Intensity_processing(single_map,norm_lib,args),norm_lib))
                            #motion_map = normalize_img(Intensity_processing(single_map))
                        # plt.imshow(motion_map)
                        # plt.show()
                #    else:
                #        pass
                        #final_output = addition(final_output,per_pyramid_eight_DSC(control_list[-1],control_list[-time_])*(decay_factor**time_))
                        #final_output = addition(final_output,eight_dir_sim(control_list[-1],control_list[-time_],args = 0)*(decay_factor**time_))
                frame_count += 1
                
                    # tr, tl, tu, td = dir_motion_processing(control_list[-time_], control_list[-1])
                    # itr, itl, itu, itd = map(Intensity_processing, [tr,tl,tu,td])
                    # norm = Normalize(vmin=0,vmax=max(np.max(itr),np.max(itl),np.max(itu),np.max(itd)))
                    # fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2)
                    # ax1.imshow(itr,norm=norm)
                    # ax1.set_title("itr")
                    # ax2.imshow(itl,norm=norm)
                    # ax2.set_title("itl")
                    # ax3.imshow(itu,norm=norm)
                    # ax3.set_title("itu")
                    # ax4.imshow(itd,norm=norm)
                    # ax4.set_title("itd")
                    # plt.suptitle(f"Time_ = {time_},frame = {frame_count}")
                    # plt.show()
                    #output_list.append(sum([normalize_img(itr),normalize_img(itl),normalize_img(itu),normalize_img(itd)]))
                # norm_max = max(list(map(np.max,output_list)))+1e-6
                # output_list = list(map(lambda x:np.uint8(250 * x / norm_max),output_list))
                # frame_output = normalize_img(sum(list(map(normalize_img,output_list))))
                # frame_output = Intensity_processing(frame_output)
                # if frame_count == 0:
                #    out_frame_list.append(frame_output/(1-gamma))
                # out_frame_list.append(frame_output)
                # out_frame_list[0] = out_frame_list[1] + out_frame_list[0] * gamma
                # out_frame_list.pop()
                #final_output = normalize_img(final_output)
                # print(final_output)
                # plt.imshow(final_output)
                # plt.show()
                #final_output = np.uint8(255 * final_output / (np.max(final_output)+1e-6))               
                #final_output = cv2.resize(final_output,final_shape)
                #final_output = cv2.cvtColor(final_output, cv2.COLOR_GRAY2BGR)
                #final_output = np.uint8(final_output//4*3+frame//4)
                #print(np.max(final_output))
                #print(np.min(final_output))
                
                #writer.write(final_output)