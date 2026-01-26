import numpy as np
import cupy as cp
import argparse
import cv2
from cupyx.scipy.ndimage import zoom, convolve1d
from utils_cupy import cpnormalize_img,cpnormalize_img3d_dict, CupyImageProcessing, CupyImagePyramid


def resize2normal(image):
    tgtw = 320
    tgth = 240
    ratio_x = tgtw/image.shape[0]
    ratio_y = tgth/image.shape[1]
    return zoom(image,[ratio_x, ratio_y])

def seperate_RGB_chanells(image):
   """
    Seperate RGB chanells to going through following algorithms
    """
   return image[:,:,2], image[:,:,1], image[:,:,0]

class CupyPreProcessor:
    def __init__(self, args, shape_hw):  # shape_hw = (H, W)
    
        gamma = args.gamma_correction
        inv_gamma = 1.0 / gamma
        self.table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)], dtype=np.uint8)
        
        H, W = shape_hw
        # Gaussian kernels: (H, 1) and (W, 1)
        k_h = cv2.getGaussianKernel(H, H / 2.0)
        k_w = cv2.getGaussianKernel(W, W / 2.0)
        k_h /= k_h.max()
        k_w /= k_w.max()
        
        # mask: (H, W)
        self.mask = (k_h @ k_w.T).astype(np.float32)  # outer product
        self.neg_mask = 1.0 - self.mask

        # Transfer to GPU once
        self.mask_gpu = cp.asarray(self.mask)
        self.neg_mask_gpu = cp.asarray(self.neg_mask)


    def process(self, image):
        """
        Args:
            image: np.ndarray of shape (H, W, C), uint8, on CPU
        
        Returns:
            cp.ndarray of shape (H, W, C), float32, on GPU
        """
        # Gamma correction on CPU (OpenCV)
        gamma_corrected = cv2.LUT(image, self.table)  # (H, W, C), uint8
        # Move to GPU and convert to float32
        img_gpu = cp.asarray(gamma_corrected, dtype=cp.float32)
        # Compute mean on GPU
        img_mean = cp.mean(img_gpu)
        # Apply mask: (H, W, 1) * (H, W, C) → (H, W, C)
        mask_expanded = self.mask_gpu[..., cp.newaxis]      # (H, W, 1)
        neg_mask_expanded = self.neg_mask_gpu[..., cp.newaxis]  # (H, W, 1)
        gaussian_img = mask_expanded * img_gpu + neg_mask_expanded * img_mean # (H, W, 3)
        return gaussian_img
    
    def reset(self):
        # build an empty ICpyrimid
        pass


class ICpyrimid:
    def __init__(self, args):
        self.args = args
        self.c_set = sorted(self.args.center)
        self.delta_set = sorted(self.args.surrounding)
        self.cs_lst = [(c,c+d) for c in self.c_set for d in self.delta_set]
        self.cal_lst = list(set([c for (c,s) in self.cs_lst] + [s for (c,s) in self.cs_lst]))
        self.total_height = args.total_height
        self.build_pyramid = CupyImagePyramid()
        self.img_process = CupyImageProcessing()
        self.transformI = cp.asarray([0.299, 0.587, 0.114])
        self.shapes = [(240, 320), (120, 160), (60, 80), (30, 40), (15, 20), (8, 10), (4, 5), (2, 3)]
        # Pyr shapes should be:
        # (240, 320)
        # (120, 160)
        # (60, 80)
        # (30, 40)
        # (15, 20)
        # (8, 10)
        # (4, 5)
        # (2, 3)
        self.ID_idx = [0,3]
        self.color_idx = [1,2,4,5]
        self.final_channel_cnt = 6
        self.reset()

    def reset(self):
        self.ICs = [cp.zeros(shape + (self.final_channel_cnt,), dtype=cp.float32) for shape in self.shapes]
        self.scaling_dict = {(c,s):cp.zeros_like(self.ICs[0]) for (c,s) in self.cs_lst}
    
    def build(self, gaussian_img):
        
        self.ICs = [cp.zeros(shape + (self.final_channel_cnt,), dtype=cp.float32) for shape in self.shapes]
        # we arrange the 6 channels as I, R, B, D, G, Y
        sum_ = cp.sum(gaussian_img, axis=-1)/2
        # print(sum_.shape)
        self.ICs[0][..., 0] = cp.tensordot(gaussian_img,self.transformI,axes=([-1],[0]))
        self.ICs[0][..., 1] = gaussian_img[...,2]*1.5 - sum_
        self.ICs[0][..., 2] = gaussian_img[...,0]*1.5 - sum_
        self.ICs[0][..., 3] = cp.max(gaussian_img) - self.ICs[0][..., 0]
        self.ICs[0][..., 4] = gaussian_img[...,1]*1.5 - sum_
        self.ICs[0][..., 5] = (gaussian_img[...,1] + gaussian_img[...,2]) / 2 - cp.abs(gaussian_img[...,1] - gaussian_img[...,2]) / 2 - gaussian_img[...,1]
        self.ICs[0] = cp.maximum(self.ICs[0],0)
        # print(Is.shape) # (240, 320)
        self.build_pyramid.eight_pyramid_built_3d(self.ICs)
        # for i in range(8):
        #     print(self.ICs[i].shape)
        #     print(cp.sum(self.ICs[i]))
    
    def scaling(self, c,s):
        diff = self.ICs[c][:,:,:3] - self.ICs[c][:,:,3:]
        diff = self.img_process.subtraction_torch(diff, self.ICs[s][:,:,:3] - self.ICs[s][:,:,3:])
        diff -= cp.mean(diff, axis=(0,1), keepdims=True)
        self.scaling_dict[(c,s)] = cp.concatenate((cp.maximum(diff, 0), cp.maximum(-diff, 0)), axis=2)
        

    def diff_process(self):
        for (c,s) in self.cs_lst:
            self.scaling(c,s)
        cpnormalize_img3d_dict(self.scaling_dict)
    
    def get_conspicuous_map(self):
        I_bar = cp.zeros((1,1))#self.ICs[0][..., 0].shape, dtype=cp.float32)  # Initialize with proper shape
        C_bar = cp.zeros((1,1))#self.ICs[0][..., 0].shape, dtype=cp.float32)  # Initialize with proper shape
        for (c,s) in self.cs_lst:
            I_bar = self.img_process.addition_torch(I_bar, cp.sum(self.scaling_dict[(c,s)][:,:,self.ID_idx], axis=-1))
            C_bar = self.img_process.addition_torch(C_bar, cp.sum(self.scaling_dict[(c,s)][:,:,self.color_idx], axis=-1))
        return I_bar, C_bar



if __name__ == '__main__':
    import matplotlib.pyplot as plt
    def parse_args():
        parse = argparse.ArgumentParser(description='Essential parameters for gabor processing') 
        parse.add_argument('--image_path', default="E:\\your_path\\RealtimeSaliency\\test_image\\circle.jpg", type=str, help='path of sample image') 
        parse.add_argument('--total_height',default=9,type=int,help='total height of orientation pyramid, equals to height of gaussian pyramid + kernel size Pyramid')
        parse.add_argument('--pyramid_height', default=5, type=int, help='height of Gaussian Pyramid')
        parse.add_argument('--gabor_kernel_size',default=33,type=int,help='the minimal value of gabor kernel size. when meet some constrains, we would double some params.')
        parse.add_argument('--num_of_thetas',default=4,type=int,help='different gabor filter orientations.')
        parse.add_argument('--mini_sigma',default=0.5,type=float,help='sigma of gabor kernel, if the image is too small hori2then double it.')
        parse.add_argument('--gabor_lambda',default=np.pi/np.sqrt(2*np.log(1/0.5)),type=float,help = 'lambda for gabor kernels')
        parse.add_argument('--gabor_gamma',default=1,type=float,help = 'gamma value for gabor filter.')
        parse.add_argument('--default_size',default=(640,480),type=tuple,help = 'default size of image')
        parse.add_argument('--center',default = (1,2),type=tuple,help = "center params. Itti default params are (2, 3, 4)")
        parse.add_argument('--surrounding',default = (3,4),type=tuple,help = "surrounding params. Itti default params are (3, 4)")
        parse.add_argument('--gamma_correction', type=float, default=2.2, help='Gamma correction value')
        args = parse.parse_args() 
        return args
    
    args = parse_args()
    
    # Load the test image from the provided path
    test_image_path = '/mnt/e/Li Lab/CVPR_rebuttal_/cuda-conv/data/checker.png'
    try:
        image = cv2.imread(test_image_path)
        if image is None:
            print(f"Could not load image from {test_image_path}, using random image instead")
            image = np.random.randint(0, 256, (240, 320, 3), dtype=np.uint8)
        else:
            # Resize image to standard size if needed
            image = cv2.resize(image, (320, 240))
            # print(f"Loaded test image with shape: {image.shape}")
    except Exception as e:
        print(f"Error loading test image: {e}, using random image instead")
        image = np.random.randint(0, 256, (240, 320, 3), dtype=np.uint8)
    
    # print(f"Using image with shape: {image.shape}")
    
    preprocessor = CupyPreProcessor(args, (240, 320))
    processed_img = preprocessor.process(image)
    
    # Create an ICpyrimid instance and test the build function
    icpyramid = ICpyrimid(args)
    icpyramid.build(processed_img)
    
    print("\n--- Pyramid Construction Completed ---")
    
    # Visualize the pyramid levels
    print("Visualizing pyramid levels...")
    pyramid_levels = min(4, len(icpyramid.ICs))  # Show first 4 levels
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    for i in range(min(6, len(icpyramid.ICs))):
        row = i // 3
        col = i % 3
        # Display the first channel of each pyramid level
        im = axes[row, col].imshow(cp.asnumpy(icpyramid.ICs[i][..., 0]), cmap='viridis')
        axes[row, col].set_title(f'Pyramid Level {i}\nShape: {icpyramid.ICs[i].shape}')
        axes[row, col].axis('off')
        plt.colorbar(im, ax=axes[row, col], fraction=0.046, pad=0.04)
    
    # Hide any unused subplots
    for i in range(pyramid_levels, 6):
        row = i // 3
        col = i % 3
        if i >= pyramid_levels:
            axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig('/mnt/e/Li Lab/CVPR_rebuttal_/BIAS-a-Biologically-Inspired-Algorithm-for-video-Saliency-detection/pyramid_levels.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Process the scaling operations
    print("\n--- Processing Scaling Operations ---")
    icpyramid.diff_process()
    
    print("Scaling completed. Visualizing scaling results...")
    
    # Visualize scaling results for the first few combinations
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Show scaling results for the first few (c,s) pairs
    cs_pairs = list(icpyramid.cs_lst)[:4]  # Take first 4 pairs
    for idx, (c, s) in enumerate(cs_pairs):
        if idx >= 4:  # Limit to 4 plots
            break
        row = idx // 2
        col = idx % 2
        
        # Show the first channel of scaling result
        scaling_result = icpyramid.scaling_dict[(c, s)]
        im = axes[row, col].imshow(cp.asnumpy(scaling_result[..., 0]), cmap='hot')
        axes[row, col].set_title(f'Scaling (c={c}, s={s})\nShape: {scaling_result.shape}')
        axes[row, col].axis('off')
        plt.colorbar(im, ax=axes[row, col], fraction=0.046, pad=0.04)
    
    # Hide any unused subplots
    for idx in range(len(cs_pairs), 4):
        row = idx // 2
        col = idx % 2
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig('/mnt/e/Li Lab/CVPR_rebuttal_/BIAS-a-Biologically-Inspired-Algorithm-for-video-Saliency-detection/scaling_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Generate and visualize the conspicuous map
    print("\n--- Generating Conspicuous Map ---")
    I_bar, C_bar = icpyramid.get_conspicuous_map()
    
    print(f"I_bar shape: {I_bar.shape}")
    print(f"C_bar shape: {C_bar.shape}")
    
    # Visualize the final results
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Original input image
    axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Original Input Image')
    axes[0].axis('off')
    
    # Intensity saliency map
    im1 = axes[1].imshow(cp.asnumpy(I_bar), cmap='hot')
    axes[1].set_title('Intensity Saliency Map (I_bar)')
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    
    # Color saliency map
    im2 = axes[2].imshow(cp.asnumpy(C_bar), cmap='hot')
    axes[2].set_title('Color Saliency Map (C_bar)')
    axes[2].axis('off')
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig('/mnt/e/Li Lab/CVPR_rebuttal_/BIAS-a-Biologically-Inspired-Algorithm-for-video-Saliency-detection/conspicuous_maps.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\nAll visualizations saved!")
    print("Pyramid levels saved as pyramid_levels.png")
    print("Scaling results saved as scaling_results.png")
    print("Conspicuous maps saved as conspicuous_maps.png")
