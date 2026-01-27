import numpy as np
import cupy as cp
import argparse
from cupyx.scipy.ndimage import convolve1d, convolve
from utils_cupy import CupyImageProcessing, cpnormalize_img, cpnormalize_img3d_dict, cpnormalize_img3d
import cv2


class CatGaborFilter:
    def __init__(self, frequency: float, theta_lst: list[float], sigma: float = None, ksize: int = 15, mode='reflect'):
        self.frequency = frequency
        self.theta_lst = cp.asarray(theta_lst, dtype=cp.float16)
        self.ksize = ksize
        self.mode = mode
        
        if sigma is None:
            sigma = 2 * 3.1415926545 / frequency
        self.sigma = sigma

        x = cp.arange(-ksize, ksize + 1, dtype=cp.float16)
        gauss = cp.exp(-x**2 / (2.0 * sigma**2))
        
        self.kernels = []
        for theta in theta_lst:
            fx = frequency * cp.cos(cp.deg2rad(theta))
            fy = frequency * cp.sin(cp.deg2rad(theta))
            
            # 先用 float32 计算，再转成 float16 存储用于卷积
            kx_real = (gauss * cp.cos(2 * cp.pi * fx * x)).astype(cp.float16)
            kx_imag = (gauss * cp.sin(2 * cp.pi * fx * x)).astype(cp.float16)
            ky_real = (gauss * cp.cos(2 * cp.pi * fy * x)).astype(cp.float16)
            ky_imag = (gauss * cp.sin(2 * cp.pi * fy * x)).astype(cp.float16)
            
            self.kernels.append({
                'kx_re': kx_real, 'kx_im': kx_imag,
                'ky_re': ky_real, 'ky_im': ky_imag
            })

    def __call__(self, image: cp.ndarray) -> cp.ndarray:
        if image.ndim == 3:
            image = image[:, :, 0]
        
        # 将输入图像转换为 float16
        # 如果图像原本是 uint8 (0-255)，建议先归一化或直接转 float16
        image = image.astype(cp.float16) / 255.0
        # image = image.astype(cp.float16) 
        
        H, W = image.shape
        C = len(self.kernels)
        
        # 预分配 float16 的输出容器
        magnitudes = cp.empty((H, W, C), dtype=cp.float16)
        
        for i, k in enumerate(self.kernels):
            # 这里的 convolve1d 会根据输入自动选择 float16 计算
            tmp_re = convolve1d(image, k['kx_re'], axis=1, mode=self.mode)
            tmp_im = convolve1d(image, k['kx_im'], axis=1, mode=self.mode)
            
            # 复数乘法逻辑： (A+Bi)(C+Di) = (AC-BD) + (AD+BC)i
            real_final = convolve1d(tmp_re, k['ky_re'], axis=0, mode=self.mode) - \
                         convolve1d(tmp_im, k['ky_im'], axis=0, mode=self.mode)
                         
            imag_final = convolve1d(tmp_re, k['ky_im'], axis=0, mode=self.mode) + \
                         convolve1d(tmp_im, k['ky_re'], axis=0, mode=self.mode)
            
            # 使用 float16 计算幅值
            magnitudes[:, :, i] = cp.sqrt(real_final**2 + imag_final**2)
            
        return magnitudes
        

class Orientation_Saliency:
    def __init__(self, args):
        # init args
        self.args = args
        self.c_set = sorted(self.args.center)
        self.delta_set = sorted(self.args.surrounding)
        self.theta_set = (0,45,90,135)
        self.O_dict = {}
        self.cs_lst = [(c,c+d) for c in self.c_set for d in self.delta_set]
        self.cal_lst = list(set([c for (c,s) in self.cs_lst] + [s for (c,s) in self.cs_lst]))
        default_freq = 2 * np.pi ** 2 / 2.7
        kernal_size = 4
        self.total_height = args.total_height
        self.pyramid_height = args.pyramid_height
        self.reset()
        # tools
        self.image_processing = CupyImageProcessing()
        self.gabor_filters = []
        for level in range(self.total_height - self.pyramid_height+1):
            self.gabor_filters.append(CatGaborFilter(frequency=default_freq, theta_lst=self.theta_set, sigma=kernal_size,ksize=kernal_size))
            default_freq /= 2
            kernal_size = round(cp.pi / default_freq) + 1

    def reset(self):
        self.Os = [None] * self.args.total_height # each of them correspond to a (H,W,C) tensor, C means channels = 'thetas'
        for c,s in self.cs_lst:
            self.O_dict[(c,s)] = None # initialize
    
    def O_c_s_theta(self,c,s):
        # theta_idx = theta // 45  # Convert angle to index (0->0, 45->1, 90->2, 135->3)
        return cp.abs(self.image_processing.subtraction_torch(self.Os[c], self.Os[s]))
    
    def Orientation_maps(self):
        for c,s in self.cs_lst:
            self.O_dict[(c,s)] = self.O_c_s_theta(c,s)
    
    def synthesis_O_map(self):
        cpnormalize_img3d_dict(self.O_dict)
        O_bars = cp.zeros((1,1,1), dtype=cp.float16) #((shape+(1)), dtype=cp.float32)
        for c, s in self.cs_lst:
            O_bars = self.image_processing.addition_torch(O_bars, self.O_dict[(c, s)])
        return cp.sum(cpnormalize_img3d(O_bars), axis=2)
    
    def build_pyramid(self, Is:list[cp.uint8]):
        for idx in range(1,self.pyramid_height):
            if idx not in self.cal_lst:
                self.Os[idx] = (cp.zeros((1,1, 1)))
            else:
                self.Os[idx] = (self.gabor_filters[0](Is[idx][:,:,:1])) # using the Is from the isaliency cupy
        for idx in range(self.pyramid_height,self.total_height):
            if idx not in self.cal_lst:
                self.Os[idx] = (cp.zeros((1,1, 1)))
            else:
                self.Os[idx] = (self.gabor_filters[idx-self.pyramid_height+1](Is[idx][:,:,:1])) # using the Is from the isaliency cupy


if __name__ == '__main__':
    from isaliency_cupy import CupyPreProcessor, ICpyrimid
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
        parse.add_argument('--default_size',default=(320,240),type=tuple,help = 'default size of image')
        parse.add_argument('--center',default = (1,),type=tuple,help = "center params. Itti default params are (2, 3, 4)")
        parse.add_argument('--surrounding',default = (4,),type=tuple,help = "surrounding params. Itti default params are (3, 4)")
        parse.add_argument('--gamma_correction', type=float, default=2.2, help='Gamma correction value')
        args = parse.parse_args() 
        return args

    args = parse_args()
    test_image_path = '/mnt/e/Li Lab/CVPR_rebuttal_/cuda-conv/data/lena.png'
    try:
        image = cv2.imread(test_image_path)
        if image is None:
            print(f"Could not load image from {test_image_path}, using random image instead")
            image = np.random.randint(0, 256, (240, 320, 3), dtype=np.uint8)
        else:
            # Resize image to standard size if needed
            image = cv2.resize(image, (320, 240))
            print(f"Loaded test image with shape: {image.shape}")
    except Exception as e:
        print(f"Error loading test image: {e}, using random image instead")
        image = np.random.randint(0, 256, (240, 320, 3), dtype=np.uint8)
    
    print(f"Using image with shape: {image.shape}")

    preprocessor = CupyPreProcessor(args, (240, 320))
    processed_img = preprocessor.process(image)
    
    # Create an ICpyrimid instance and test the build function
    icpyramid = ICpyrimid(args)
    icpyramid.build(processed_img)
    
    print("\n--- Building Orientation Saliency ---")
    Ori_processing = Orientation_Saliency(args)
    Ori_processing.build_pyramid(icpyramid.ICs)
    
    print("Orientation pyramid built. Visualizing different orientation filters...")
    
    # Visualize different orientation Gabor filters applied to the base image
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    orientations = [0, 45, 90, 135]
    
    base_image = icpyramid.ICs[0][..., :1]  # Use the first channel of the base level
    
    for idx, theta in enumerate(orientations):
        row = idx // 2
        col = idx % 2
        
        # Apply the corresponding Gabor filter
        gabor_filter = Ori_processing.gabor_filters[0] # First level filters
        filtered_result = gabor_filter(base_image)[:,:,idx] 
        
        im = axes[row, col].imshow(cp.asnumpy(filtered_result), cmap='gray')
        axes[row, col].set_title(f'Gabor Filter: {theta}°\nShape: {filtered_result.shape}')
        axes[row, col].axis('off')
        plt.colorbar(im, ax=axes[row, col], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig('/mnt/e/Li Lab/CVPR_rebuttal_/BIAS-a-Biologically-Inspired-Algorithm-for-video-Saliency-detection/orientation_gabor_filters.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("Computing orientation contrast maps...")
    # Compute orientation contrast maps
    Ori_processing.Orientation_maps()
    
    print("Visualizing orientation contrast maps...")
    
    # Visualize some of the orientation contrast maps (O_dict contents)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Show some (c,s) combinations
    cs_combinations = [(c, s) for c, s in Ori_processing.cs_lst[:2]]  # Limit to first few
    for (c, s) in cs_combinations[:4]:  # Show first 4
        for idx in range(4):
            row = idx // 2
            col = idx % 2

            if (c, s) in Ori_processing.O_dict and Ori_processing.O_dict[(c, s)] is not None:
                result = Ori_processing.O_dict[(c, s)]
                im = axes[row, col].imshow(cp.asnumpy(result[..., idx]), cmap='hot')
                axes[row, col].set_title(f'O_dict[{c},{s}]\nShape: {result.shape}')
                axes[row, col].axis('off')
                plt.colorbar(im, ax=axes[row, col], fraction=0.046, pad=0.04)
            else:
                axes[row, col].text(0.5, 0.5, f'Missing:\n({c},{s})', 
                                   horizontalalignment='center', verticalalignment='center',
                                   transform=axes[row, col].transAxes, fontsize=12)
                axes[row, col].axis('off')
    
    
    plt.tight_layout()
    plt.savefig('/mnt/e/Li Lab/CVPR_rebuttal_/BIAS-a-Biologically-Inspired-Algorithm-for-video-Saliency-detection/orientation_contrast_maps.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("Generating final orientation saliency map...")
    # Generate and visualize the final orientation saliency map
    final_osaliency = Ori_processing.synthesis_O_map()
    
    print(f"Final orientation saliency map shape: {final_osaliency.shape}")
    
    # Final visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Original input image
    axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Original Input Image')
    axes[0].axis('off')
    
    # Base intensity layer for reference
    base_intensity = cp.asnumpy(icpyramid.ICs[0][..., 0])
    im1 = axes[1].imshow(base_intensity, cmap='gray')
    axes[1].set_title('Base Intensity Layer')
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    
    # Final orientation saliency map
    im2 = axes[2].imshow(cp.asnumpy(final_osaliency), cmap='hot')
    axes[2].set_title('Final Orientation Saliency Map')
    axes[2].axis('off')
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig('/mnt/e/Li Lab/CVPR_rebuttal_/BIAS-a-Biologically-Inspired-Algorithm-for-video-Saliency-detection/final_orientation_saliency.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\nAll orientation saliency visualizations saved!")
    print("Gabor filters saved as orientation_gabor_filters.png")
    print("Orientation contrast maps saved as orientation_contrast_maps.png")
    print("Final orientation saliency map saved as final_orientation_saliency.png")
