import os
import cv2
import numpy as np
import tqdm

if __name__ == "__main__":
    video_path = r'e:\Li Lab\itti_and_lif\Realtime_result\Levy'
    tgt_path = r'e:\Li Lab\itti_and_lif\VAL_results\Levy'
    if not os.path.exists(tgt_path): # create folder
        os.mkdir(tgt_path)
    # true_vid_path = r'E:\Li Lab\itti_and_lif\video\annotation'
    # for filename in os.listdir(video_path):
    #     if filename.endswith(".mp4"):
    #         vid_num = int(filename.split('.')[0])
    #         # print(f'{vid_num:04d}')
    #         this_video_path = os.path.join(video_path, filename)
    #         gtvideo_path = os.path.join(true_vid_path, f'{vid_num:04d}','maps')
    #         gt_video_length = len(os.listdir(gtvideo_path))
    #         # load video
    #         cap = cv2.VideoCapture(this_video_path)
    #         frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    #         # print([this_video_path, gtvideo_path])
    #         # print([frame_count, gt_video_length])
    #         assert frame_count == gt_video_length
    #         cap.release()
    for filename in tqdm.tqdm(os.listdir(video_path)):
        if filename.endswith(".mp4"):    
            vid_num = filename.split('.')[0] # int(filename.split('.')[0])
            this_video_path = os.path.join(video_path, filename)
            cap = cv2.VideoCapture(this_video_path)

            tgt_img_paths = os.path.join(tgt_path, vid_num ) # f'{vid_num:04d}'
            os.mkdir(tgt_img_paths)
            for i in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
                ret, frame = cap.read()
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    # frame = cv2.resize(frame, (256, 256))
                    cv2.imwrite(os.path.join(tgt_img_paths, f'{1+i:04d}.png'), frame)
                else:
                    break

