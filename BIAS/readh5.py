import h5py
import numpy as np
import matplotlib.pyplot as plt


with h5py.File('h5rst\\image_saliency_map.h5','r') as f:
    print(list(f.keys()))
    print(len(list(f.keys())))
    image_saliency_lst = [f[list(f.keys())[x]].shape for x in range(len(list(f.keys())))]
    for i in range(len(list(f.keys()))):
        sample = np.float64(f[list(f.keys())[i]][:,:,1])
        plt.imshow(sample,cmap='gray')
        plt.show()
        if i>= 0:
            break

with h5py.File('h5rst\\motion_saliency_map.h5','r') as f:
    print(list(f.keys()))
    print(len(list(f.keys())))
    motion_saliency_lst = [f[list(f.keys())[x]].shape for x in range(len(list(f.keys())))]
    for i in range(len(list(f.keys()))):
        sample = np.float64(f[list(f.keys())[i]][:,:,1])
        plt.imshow(sample,cmap='gray')
        plt.show()
        if i>= 0:
            break

for i in range(len(image_saliency_lst)):
    if image_saliency_lst[i] != motion_saliency_lst[i]:
        print(f'error{i}')
        break
else:
    print("checked ok")
        
