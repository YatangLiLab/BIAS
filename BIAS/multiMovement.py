import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
from tqdm import trange
from utils import *
from image_saliency import Is_scale
import time

class Camera():
    def __init__(self):
        self.mov_dict = {"hori":np.zeros((4,4)),"vert":np.zeros((4,4))}
        self.delta_t = 1/10 # the update factor of update value
    def initialize(self,motion_lsts=[]):
        self.mov_dict["hori"] = np.zeros((4,4))
        self.mov_dict['vert'] = np.zeros((4,4))
    def update(self,hori_value,vert_value):
        # negative for leftward and upward, positive for rightward and downward
        # should be put in as a series of numbers?
        self.mov_dict["hori"] += self.delta_t * (hori_value - self.mov_dict["hori"])
        self.mov_dict["vert"] += self.delta_t * (vert_value - self.mov_dict["vert"])
        #print(self.mov_dict)
        # update as v(t+1) = v(t) + (target - v(t))dt

def simple_minus_separation(mat_lst1,mat_lst2):
    """
    simply use minus to find the true value of each point optic flow, not after calculating center-surrounding
    """
    assert len(mat_lst1) == len(mat_lst2)
    tmp_mat_lst = [mat_lst1[i]-mat_lst2[i] for i in range(len(mat_lst1))]
    return [np.where(tmp_mat_lst[i]>0,tmp_mat_lst[i],0) for i in range(len(tmp_mat_lst))],[np.where(tmp_mat_lst[i]<0, - tmp_mat_lst[i],0) for i in range(len(tmp_mat_lst))]

def sum_of_image_lst(mat_lst):
    """
    calculate the sum of a matrix list, after some resize.
    """
    rst_image = np.zeros((1,1))
    for i in range(len(mat_lst)):
        rst_image = addition(rst_image, mat_lst[i])
    return rst_image

def non_linear(x):
    return x/(np.max(x)/2+1e-3)/(np.sqrt(1+x**2/(np.max(x)**2/4+1e-3))+1e-3)

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

    c_set = args.motion_center
    delta_set = args.motion_surrounding
    I_dict = {}
    for c in c_set:
        for delta in delta_set:
            I_dict[(c,c+delta)] = Is_scale(Is,Ds,c,c+delta)
    I_bar = np.zeros((1,1),dtype=np.float32)
    for c in c_set:
        for delta in delta_set:
            assert I_dict[(c,c+delta)][0].shape == I_dict[(c,c+delta)][1].shape
            I_bar = addition(I_bar,I_dict[(c,c+delta)][0])#normalize_img(np.float32(I_dict[(c,c+delta)][0]),norm_lib))  # White
            I_bar = addition(I_bar,I_dict[(c,c+delta)][1])#normalize_img(np.float32(I_dict[(c,c+delta)][1]),norm_lib))  # Black
            # darkness term# 
    if ifshow:
        fig, axs = plt.subplots(2 * len(c_set) + 1,max(len(delta_set),2))
        axs[0,0].imshow(image)
        axs[0,0].set_title("original image")
        axs[0,1].imshow(I_bar)#normalize_img(I_bar,norm_lib))
        axs[0,1].set_title("output image")
        for i in range(len(c_set)):
            for j in range(len(delta_set)):
                axs[1+2*i,j].imshow(I_dict[(c_set[i],c_set[i]+delta_set[j])][0])
                axs[1+2*i,j].set_title(f"I_dict[{(c_set[i],c_set[i]+delta_set[j])}][0]")
                axs[2+2*i,j].imshow(I_dict[(c_set[i],c_set[i]+delta_set[j])][1])
                axs[2+2*i,j].set_title(f"I_dict[{(c_set[i],c_set[i]+delta_set[j])}][1]")

        # # plt.imshow(I_bar)
        # fig, axs = plt.subplots(2,2)
        # axs[0,0].imshow(image)
        # axs[0,1].imshow(normalize_img(I_bar,norm_lib))
        # axs[1,0].imshow(Is[0])
        # axs[1,1].imshow(Ds[0])
        plt.show()
    return I_bar

def four_dir_sim(Is0,Is1,args,current_camera:Camera,ifshow = False)->np.array:
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
    motion_list = ["ds","ls","us","rs"]
    current_velocity = [current_camera.mov_dict['hori'],current_camera.mov_dict['vert']]
    results = []
    half = len(directions)//2
    # static_map = np.ones_like((1,1))
    hori_value = np.zeros((4,4))
    vert_value = np.zeros((4,4))
    for spatial_index in (1,2,3,4):
        current_frame = np.int16(Is0[spatial_index])
        former_frame = np.int16(Is1[spatial_index])
        max_form = np.where(current_frame>former_frame,current_frame,former_frame)
        tmp = np.abs(current_frame - former_frame)
        # tmp = cv2.GaussianBlur(np.abs(current_frame - former_frame),(3,3),sigmaX= 0.02 * current_frame.shape[0]) / (max_form+1e-3)
        #tmp = np.where(tmp>np.percentile(tmp,75),1,0)
        # plt.imshow(tmp)
        # plt.show())
        #if ifshow:
        #    fig,((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2)
        #    ax1.imshow(current_frame/(max_form+25))
        #    ax2.imshow(former_frame/(max_form+25))
        #    ax3.imshow(-np.abs(current_frame/(max_form+25) - former_frame/(max_form+25)))
        #    ax4.imshow(stable_img)
        #    plt.show()
        #plt.imshow(stable_img)
        #plt.show()
        shape = current_frame.shape
        edge = 16
        padded_former_frame = np.pad(former_frame,edge,mode='reflect')
        x_Quartiles = [0,shape[0]//4,shape[0]//2,shape[0]*3//4,shape[0]]
        y_Quartiles = [0,shape[1]//4,shape[1]//2,shape[1]*3//4,shape[1]]
        for dir_pair_index in range(half):
            adjust = [current_velocity[1]//(2**spatial_index),current_velocity[0]//(2**spatial_index)]
            l_pair = np.zeros_like(current_frame)
            r_pair = np.zeros_like(current_frame)
            for x_index in range(4):
                for y_index in range(4):
                    x_start = x_Quartiles[x_index]
                    x_end = x_Quartiles[x_index+1]
                    y_start = y_Quartiles[y_index]
                    y_end = y_Quartiles[y_index+1]
                    limited_x = min(edge-1,max(-edge+1,int(adjust[0][x_index,y_index]))) # in [-edge,edge]
                    limited_y = min(edge-1,max(-edge+1,int(adjust[1][x_index,y_index])))
                    l_pair[x_start:x_end, y_start:y_end] = np.abs(current_frame[x_start:x_end, y_start:y_end]
                                                        - padded_former_frame[x_start+directions[dir_pair_index][0]+edge + limited_x \
                                                                            : x_end + directions[dir_pair_index][0]+edge + limited_x ,\
                                                                              y_start + directions[dir_pair_index][1]+edge + limited_y  \
                                                                            : y_end + directions[dir_pair_index][1]+edge + limited_y ])#/(max_form[x_start:x_end, y_start:y_end]+1e-2)
                    r_pair[x_start:x_end, y_start:y_end] = np.abs(current_frame[x_start:x_end, y_start:y_end] \
                                                        - padded_former_frame[x_start+directions[half + dir_pair_index][0]+edge + limited_x  \
                                                                            : x_end + directions[half + dir_pair_index][0]+edge + limited_x ,\
                                                                              y_start + directions[half + dir_pair_index][1]+edge + limited_y  \
                                                                            : y_end + directions[half + dir_pair_index][1]+edge + limited_y])# /(max_form[x_start:x_end, y_start:y_end]+1e-2)
            # adjust = [0,0]
            # threshold = 0.01
            # 0 for horizental and 1 for vertical
            #l_pair = np.abs(current_frame - padded_former_frame[directions[dir_pair_index][0]+edge + adjust[0] : directions[dir_pair_index][0]+edge+shape[0]+ adjust[0],\
            #                                                    directions[dir_pair_index][1]+edge + adjust[1] : directions[dir_pair_index][1]+edge+shape[1]+ adjust[1]]) /(max_form+1e-7)
            #r_pair = np.abs(current_frame - padded_former_frame[directions[half + dir_pair_index][0]+edge + adjust[0] : directions[half + dir_pair_index][0]+edge+shape[0] + adjust[0],\
            #                                                    directions[half + dir_pair_index][1]+edge + adjust[1] : directions[half + dir_pair_index][1]+edge+shape[1] + adjust[1]]) /(max_form+1e-7)
            padd_r_pair = np.pad(r_pair,1,mode='reflect')
            l, r = np.exp(-1/50 * l_pair) * tmp, np.exp(-1/50 * padd_r_pair[directions[dir_pair_index][0]+1 : directions[dir_pair_index][0]+1+shape[0],\
                                                                                      directions[dir_pair_index][1]+1 : directions[dir_pair_index][1]+1+shape[1]])*tmp
            # l, r = cv2.GaussianBlur(l,(3,3),2), cv2.GaussianBlur(r,(3,3),2)
            #lr_mean = (np.mean(l)+np.mean(r))/2
            #l, r = l/(lr_mean+1e-7), r/(lr_mean+1e-7)
            l, r = np.where(l>0,l,0),np.where(r>0,r,0)
            #stable_img -= np.abs(l-r) * np.mean(current_frame)
            motion_dict[motion_list[dir_pair_index]].append(zero_edge(l))
            motion_dict[motion_list[half + dir_pair_index]].append(zero_edge(r))
        # shape_value = np.shape(motion_dict['ls'][-1])[0] * np.shape(motion_dict['ls'][-1])[1] 
        for x_index in range(4):
            for y_index in range(4):
                x_start = x_Quartiles[x_index]
                x_end = x_Quartiles[x_index+1]
                y_start = y_Quartiles[y_index]
                y_end = y_Quartiles[y_index+1]
                hori_value[x_index,y_index] += (np.mean(motion_dict['ls'][-1][x_start:x_end,y_start:y_end]) - np.mean(motion_dict['rs'][-1][x_start:x_end,y_start:y_end]))*2 ** spatial_index
                vert_value[x_index,y_index] += (np.mean(motion_dict['us'][-1][x_start:x_end,y_start:y_end]) - np.mean(motion_dict['ds'][-1][x_start:x_end,y_start:y_end]))*2 ** spatial_index
    tus, tds = simple_minus_separation(motion_dict['us'],motion_dict['ds'])
    tls, trs = simple_minus_separation(motion_dict['ls'],motion_dict['rs'])
    tu = sum_of_image_lst(tus)
    td = sum_of_image_lst(tds)
    tl = sum_of_image_lst(tls)
    tr = sum_of_image_lst(trs)
    tu = cv2.GaussianBlur(non_linear(tu),(3,3),2)
    td = cv2.GaussianBlur(non_linear(td),(3,3),2)
    tl = cv2.GaussianBlur(non_linear(tl),(3,3),2)
    tr = cv2.GaussianBlur(non_linear(tr),(3,3),2)
    not_stable = (tu)**2+(td)**2 + (tl**2)+(tr)**2 #cv2.GaussianBlur(non_linear(addition(not_stable0,not_stable1)),(3,3),2)
    current_camera.update(hori_value,vert_value)
    if ifshow:
        norm = Normalize(vmin = 0, vmax = max(np.max(tu),np.max(td),np.max(tl),np.max(tr)))
        fig, ((ax1,ax2,ax5,im1),(ax3,ax4,ax6,im2),(bx1,bx2,bx3,bx4),(bx5,bx6,bx7,bx8)) = plt.subplots(4,4)
        ax1.imshow(tr, cmap='viridis')#, norm=norm)
        ax1.set_title("tr")
        ax2.imshow(tl, cmap='viridis')#, norm=norm)
        ax2.set_title("tl")
        ax3.imshow(tu, cmap='viridis')#, norm=norm)
        ax3.set_title("tu")
        ax4.imshow(td, cmap='viridis')#, norm=norm)
        # ax4.imshow(dy_image_d)
        ax4.set_title("td")
        ax6.imshow(not_stable)
        im1.imshow(Is0[1])
        im1.set_title("Is0[1]")
        im2.imshow(Is1[1])
        im2.set_title("Is1[1]")

        bx1.imshow(motion_dict['rs'][0], cmap='viridis')#, norm=norm)
        bx1.set_title("motion r0")
        bx2.imshow(motion_dict['ls'][0], cmap='viridis')#, norm=norm)
        bx2.set_title("motion l0")
        bx3.imshow(motion_dict['us'][0], cmap='viridis')#, norm=norm)
        bx3.set_title("motion u0")
        bx4.imshow(motion_dict['ds'][0], cmap='viridis')#, norm=norm)
        bx4.set_title("motion d0")
        bx5.imshow(motion_dict['rs'][2], cmap='viridis')#, norm=norm)
        bx5.set_title("motion r1")
        bx6.imshow(motion_dict['ls'][2], cmap='viridis')#, norm=norm)
        bx6.set_title("motion l1")
        bx7.imshow(motion_dict['us'][2], cmap='viridis')#, norm=norm)
        bx7.set_title("motion u1")
        bx8.imshow(motion_dict['ds'][2], cmap='viridis')#, norm=norm)
        bx8.set_title("motion d1")
        plt.show()
    results = [tu,td,tl,tr, not_stable]
    return results


def parse_args():
    """
    get args.
    """
    parse = argparse.ArgumentParser(description='essential parameters') 
    parse.add_argument('--video_path', default="E:\\Li Lab\\itti_and_lif\\RealtimeSaliency\\test_video\\247.avi", type=str, help='path of sample video') 
    parse.add_argument('--generate_name',default = "result_video\\247-zooming-24816.mp4",type=str,help = 'default generate name')
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
    parse.add_argument('--default_checkpoint', default=(2,4,8,16),type=tuple,help = 'default checkpoints of control length')
    parse.add_argument('--selective_threshold',default = 0,type=float,help = 'default selective threshold percentage')
    parse.add_argument('--double_height',default = False,type=bool,help = 'default output window')
    parse.add_argument('--alpha_beta',default = (0,1),type=tuple,help = 'determine standard alpha and beta.')
    parse.add_argument('--output',default="write",type = str, help = "determine output format.")
    parse.add_argument('--target_folder',default='lower_result\\tests',type=str,help = 'default place to place pred results')
    parse.add_argument('--center',default = (1,2),type=tuple,help = "center params. Itti default params are (2, 3, 4)")
    parse.add_argument('--surrounding',default = (2,),type=tuple,help = "surrounding params. Itti default params are (3, 4)")
    parse.add_argument('--motion_center',default = (1,2,3),type=tuple,help = "center params. Itti default params are (2, 3, 4)")
    parse.add_argument('--motion_surrounding',default = (3,4),type=tuple,help = "surrounding params. Itti default params are (3, 4)")
    parse.add_argument('--decay_factor',default = 0.95,type=float,help = "decay factor of time-related problems.")
    parse.add_argument('--gamma_correlation',default=2.2,type = float, help = "gamma value for gamma correlation")
    parse.add_argument('--tmp_save_factor',default=0.9,type = float, help = "gamma value for gamma correlation")
    parse.add_argument('--continuity',type=bool, default=False,help="control how to output, using simple output or continue output.")
    parse.add_argument('--add_central_gaussian',type = bool, default= False, help = "control if apply central gaussian in the end")
    parse.add_argument('--output_gamma_correlation',type = bool, default= False, help = "control if apply central gaussian in the end")
    args = parse.parse_args() 
    return args

if __name__ == "__main__":
    fillIn_lib = initialize_fillIn_lib()
    tmp_save_diff = np.zeros((1,1))
    args = parse_args()
    #video_path = "E:\\Li Lab\\itti_and_lif\\SAVAM\\Videos\\v01_Hugo_2172_left.avi"
    video_path = args.video_path# "E:\\Li Lab\\itti_and_lif\\RealtimeSaliency\\test_video\\001.avi"
    # video_path = "E:\\Li Lab\\itti_and_lif\\RealtimeSaliency\\test_video\\test_motion.mp4"
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
    writer = cv2.VideoWriter(args.generate_name,fourcc,fps,final_shape)
    # writer = cv2.VideoWriter(f"notsample009g095-outputgamma-relativemotion.mp4",fourcc,fps,final_shape)
    frame_count = 0
    control_list = []
    gamma = 0.95
    decay_factor = 0.9
    out_frame_list = [] #store gamma-out image
    time_list = args.default_checkpoint

    camera_motion_lst = [Camera()] * len(args.default_checkpoint)
    camera_moving_lst = []
    # time_list = (2,4)
    if video_capture.isOpened():
        for _ in trange(int(frame_number)): # frame_number
            
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
                    results = four_dir_sim(control_list[0],control_list[checkpoint-1],args,camera_motion_lst[index])
                    grids = results[0].shape[1] * results[0].shape[0]
                    # print(f"nonzeros are left {np.count_nonzero(results[2])}, right {np.count_nonzero(results[3])}, up{np.count_nonzero(results[0])}, down{np.count_nonzero(results[1])}")
                    # camera_motion_lst[index].update(hori_value/grids,vert_value/grids)
                    #print(f"camara status:{camera_motion_lst[index].mov_dict['hori']},{camera_motion_lst[index].mov_dict['vert']}")
                    camera_moving_lst.append((camera_motion_lst[index].mov_dict['hori'],camera_motion_lst[index].mov_dict['vert']))

                    motion_saliency_map = np.zeros((1,1))
                    for single_map_index in range(len(results)):
                        single_map = results[single_map_index]
                        # single_map = fill_blank(fillIn_lib,single_map,cv2.resize(current_frame,(single_map.shape[1],single_map.shape[0]),interpolation=cv2.INTER_NEAREST))
                        motion_saliency_map = addition(motion_saliency_map,Intensity_processing(single_map,norm_lib,args)**2)
                        maps.append(motion_saliency_map * args.decay_factor ** index)
            # diff_time_scale_map = addition(diff_time_scale_map,normalize_img(eight_dir_processing(control_list[0],control_list[checkpoint])))
                diff_time_scale_map = sum(maps)
                diff_time_scale_map = normalize_img(diff_time_scale_map,norm_lib) 
                # diff_time_scale_map = fill_blank(fillIn_lib,diff_time_scale_map,cv2.resize(current_frame,(diff_time_scale_map.shape[1],diff_time_scale_map.shape[0]),interpolation=cv2.INTER_NEAREST))
                diff_time_scale_map = non_linear(diff_time_scale_map)
                diff_time_scale_map = (diff_time_scale_map - np.min(diff_time_scale_map))/(np.max(diff_time_scale_map) - np.min(diff_time_scale_map)+1e-7) * 255 if np.max(diff_time_scale_map) >0 else diff_time_scale_map
                # diff_time_scale_map = np.where(diff_time_scale_map>np.percentile(diff_time_scale_map,90),diff_time_scale_map,0)
                output_diff = diff_time_scale_map if np.max(diff_time_scale_map) > np.max(tmp_save_diff) else tmp_save_diff
                # tmp save technique
                # tmp_save_diff = output_diff if np.max(diff_time_scale_map) > np.max(tmp_save_diff) else tmp_save_diff * 0.9
                #plt.imshow(diff_time_scale_map)
                #plt.show()
                output = cv2.resize(np.uint8(output_diff),final_shape,interpolation=cv2.INTER_NEAREST)
                output = cv2.cvtColor(output,cv2.COLOR_GRAY2BGR)
                #plt.imshow(np.float32(output)*3/4/255 + np.float32(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))/4/255)
                #plt.show()
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
                
                writer.write(output//4 * 3 + frame//4)
#    plt.scatter(np.array(camera_moving_lst)[:,0],np.array(camera_moving_lst)[:,1],c="red")
#    plt.scatter(np.array(camera_moving_lst)[:,0],np.array(camera_moving_lst)[:,1],c="blue")
#    plt.plot(np.array(camera_moving_lst)[:,0],np.array(camera_moving_lst)[:,1],c="red")
#    plt.plot(np.array(camera_moving_lst)[:,0],np.array(camera_moving_lst)[:,1],c="blue")
#    id = video_path.split('\\')[-1].split(".")[0]
#    plt.title(f"camera velocity for video {id}")
#    plt.show()