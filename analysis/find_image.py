import numpy
import cv2
import os
def find_image(frame_idx, video_idx,path = 'E:\\Li Lab\\itti_and_lif\\video\\video\\video'):
    video_path = path + '\\'+ f'{video_idx:03d}' + '.avi'
    video = cv2.VideoCapture(video_path)
    idx = 0
    while True:
        ret, frame = video.read()
        if ret == False:
            raise RuntimeError('video end')
        if frame_idx == idx:
            video.release()
            return frame
        idx += 1
        
def find_gt(frame_idx, video_idx,path = 'E:\\Li Lab\\itti_and_lif\\video\\annotation'):
    gt_path = os.path.join(path, f'{video_idx:04d}','maps', f'{frame_idx:04d}')+'.png'
    gt_frame = cv2.imread(gt_path)
    return gt_frame

def find_pred(frame_idx, video_idx,path):
    pred_path = os.path.join(path, f'{video_idx:04d}', f'{frame_idx:04d}')+'.png'
    pred_frame = cv2.imread(pred_path)
    return pred_frame

if __name__ == '__main__':
    mode = input('mode: \n')
    if mode[0] in ['B','b']:
        mode = 'best'
    elif mode[0] in ['W','w']:
        mode = 'worst'
    else:
        raise RuntimeError(f'wrong mode: {mode}')
    print(mode)
    model = input('model: \n')
    model_path = 'E:\\Li Lab\\itti_and_lif\\VAL_results\\' + model
    os.mkdir(rst_folder := f'best_and_worst\\{model}')
    for i in range(5):  
        video_idx = int(input('video_idx: \n'))
        frame_idx = int(input('frame_idx: \n'))
        frame = find_image(frame_idx, video_idx)
        gt_frame = find_gt(frame_idx+1, video_idx)
        pred_frame = find_pred(frame_idx+1, video_idx, model_path)
        cv2.imwrite(f'{rst_folder}\\{mode}_{video_idx}_{frame_idx}_frame.png', frame)
        cv2.imwrite(f'{rst_folder}\\{mode}_{video_idx}_{frame_idx}_gt.png', gt_frame)
        cv2.imwrite(f'{rst_folder}\\{mode}_{video_idx}_{frame_idx}_pred.png', pred_frame)
