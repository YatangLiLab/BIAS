import numpy as np
import cupy as cp
import argparse
from cupyx.scipy.ndimage import convolve1d
from utils_cupy import CupyImageProcessing, cpnormalize_img
import cv2

class GaborFilter:
    def __init__(self, frequency: float, theta: float, sigma: float = None, ksize: int = 15, mode='reflect'):
        """
        Precompute separable Gabor filter components for efficient reuse.
        
        Args:
            frequency: spatial frequency (cycles per pixel)
            theta: orientation in radians
            sigma: standard deviation of Gaussian envelope; if None, set to 2π / frequency
            ksize: half-kernel size (full kernel = 2*ksize + 1)
            mode: boundary mode for convolve1d ('constant', 'reflect', etc.)
        """
        self.frequency = frequency
        self.theta = theta
        self.ksize = ksize
        self.mode = mode
        
        if sigma is None:
            sigma = 2 * cp.pi / frequency  # heuristic from literature
        self.sigma = sigma

        # Build 1D coordinate
        x = cp.arange(-ksize, ksize + 1, dtype=cp.float32)
        
        # Gaussian envelope
        gauss = cp.exp(-x**2 / (2.0 * sigma**2))
        
        # Projected frequency along filter axis
        freq_x = frequency * cp.cos(theta)
        freq_y = frequency * cp.sin(theta)
        
        # Real and imaginary parts of 1D Gabor kernels
        self.kernel_x_real = (gauss * cp.cos(2 * cp.pi * freq_x * x)).astype(cp.float32)
        self.kernel_x_imag = (gauss * cp.sin(2 * cp.pi * freq_x * x)).astype(cp.float32)
        
        self.kernel_y_real = (gauss * cp.cos(2 * cp.pi * freq_y * x)).astype(cp.float32)
        self.kernel_y_imag = (gauss * cp.sin(2 * cp.pi * freq_y * x)).astype(cp.float32)

    def __call__(self, image: cp.ndarray) -> cp.ndarray:
        """
        Apply Gabor filter to a 2D image (H, W).
        
        Returns:
            Magnitude response: sqrt(real^2 + imag^2), shape (H, W)
        """
        assert image.ndim == 2, "Input must be 2D (H, W)"
        image = image.astype(cp.float32)

        # Step 1: Filter along x (columns) → intermediate complex response
        real_x = convolve1d(image, self.kernel_x_real, axis=1, mode=self.mode)
        imag_x = convolve1d(image, self.kernel_x_imag, axis=1, mode=self.mode)

        # Step 2: Filter real_x and imag_x along y (rows)
        real_final = convolve1d(real_x, self.kernel_y_real, axis=0, mode=self.mode) \
                   - convolve1d(imag_x, self.kernel_y_imag, axis=0, mode=self.mode)
        imag_final = convolve1d(real_x, self.kernel_y_imag, axis=0, mode=self.mode) \
                   + convolve1d(imag_x, self.kernel_y_real, axis=0, mode=self.mode)

        # Step 3: Compute magnitude
        magnitude = cp.sqrt(real_final * real_final + imag_final * imag_final)
        return magnitude

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
            self.gabor_filters.append([GaborFilter(frequency=default_freq, theta=theta/180*cp.pi, sigma=kernal_size,ksize=kernal_size) for theta in self.theta_set])
            default_freq /= 2
            kernal_size = round(cp.pi / default_freq) + 1

    def reset(self):
        self.Os = [[None] * self.args.total_height for _ in self.theta_set]
        for c,s in self.cs_lst:
            for theta in self.theta_set:
                self.O_dict[(c,s,theta)] = None # initialize
    
    def O_c_s_theta(self,c,s,theta):
        theta_idx = theta // 45  # Convert angle to index (0->0, 45->1, 90->2, 135->3)
        return cp.abs(self.image_processing.subtraction_torch(self.Os[theta_idx][c], self.Os[theta_idx][s]))
    
    def Orientation_maps(self,ifshow=False):
        for c,s in self.cs_lst:
            for theta in self.theta_set:
                self.O_dict[(c,s,theta)] = self.O_c_s_theta(c,s,theta)
        if ifshow:
            raise NotImplementedError('not implement if show methods now')
    
    def synthesis_O_map(self, ifshow=False):
        # Initialize with the proper shape from the first element
        first_key = list(self.O_dict.keys())[0]  # Get first key to get the shape
        if self.O_dict[first_key] is not None:
            shape = self.O_dict[first_key].shape[:2]  # Get spatial dimensions
        else:
            # Fallback to a default shape if no elements exist
            shape = (240, 320)  # Default shape
        
        O_bar_0 =  cp.zeros((1,1))#(shape, dtype=cp.float32)
        O_bar_45 = cp.zeros((1,1))#(shape, dtype=cp.float32)
        O_bar_90 = cp.zeros((1,1))#(shape, dtype=cp.float32)
        O_bar_135= cp.zeros((1,1))#(shape, dtype=cp.float32)
        
        for c, s in self.cs_lst:
            # if (c, s, 0) in self.O_dict and self.O_dict[(c, s, 0)] is not None:
            O_bar_0 = self.image_processing.addition_torch(O_bar_0, cpnormalize_img(self.O_dict[(c, s, 0)]))
            # if (c, s, 45) in self.O_dict and self.O_dict[(c, s, 45)] is not None:
            O_bar_45 = self.image_processing.addition_torch(O_bar_45, cpnormalize_img(self.O_dict[(c, s, 45)]))
            # if (c, s, 90) in self.O_dict and self.O_dict[(c, s, 90)] is not None:
            O_bar_90 = self.image_processing.addition_torch(O_bar_90, cpnormalize_img(self.O_dict[(c, s, 90)]))
            # if (c, s, 135) in self.O_dict and self.O_dict[(c, s, 135)] is not None:
            O_bar_135 = self.image_processing.addition_torch(O_bar_135, cpnormalize_img(self.O_dict[(c, s, 135)]))
            print('=======================================')
            print(cp.sum(O_bar_0),cp.sum(O_bar_45),cp.sum(O_bar_90),cp.sum(O_bar_135))
            print(cp.sum(self.O_dict[(c, s, 0)]  - cp.min(self.O_dict[(c,s,0  )])),
                 cp.sum(self.O_dict[(c, s, 45)]  - cp.min(self.O_dict[(c,s,45 )])),
                 cp.sum(self.O_dict[(c, s, 90)]  - cp.min(self.O_dict[(c,s,90 )])),
                 cp.sum(self.O_dict[(c, s, 135)] - cp.min(self.O_dict[(c,s,135)])))
            print('=======================================')
        return sum(map(cpnormalize_img, [O_bar_0, O_bar_45, O_bar_90, O_bar_135]))
    
    def build_pyramid(self, Is:list[cp.uint8]):
        for idx in range(1,self.pyramid_height):
            if idx not in self.cal_lst:
                for _ in range(len(self.theta_set)):
                    self.Os[_][idx] = (cp.zeros((1,1)))
            else:
                for _ in range(len(self.theta_set)):
                    self.Os[_][idx] = (self.gabor_filters[0][_](Is[idx][:,:,0].copy())) # using the Is from the isaliency cupy
        for idx in range(self.pyramid_height,self.total_height):
            if idx not in self.cal_lst:
                for _ in range(len(self.theta_set)):
                    self.Os[_][idx] = (cp.zeros((1,1)))
            else:
                for _ in range(len(self.theta_set)):
                    self.Os[_][idx] = (self.gabor_filters[idx-self.pyramid_height+1][_](Is[idx][:,:,0].copy())) # using the Is from the isaliency cupy


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
    
    base_image = icpyramid.ICs[0][..., 0]  # Use the first channel of the base level
    
    for idx, theta in enumerate(orientations):
        row = idx // 2
        col = idx % 2
        
        # Apply the corresponding Gabor filter
        gabor_filter = Ori_processing.gabor_filters[0][idx]  # First level filters
        filtered_result = gabor_filter(base_image)
        
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
    
    # Show some (c,s,theta) combinations
    cs_theta_combinations = [(c, s, theta) for c, s in Ori_processing.cs_lst[:2] for theta in orientations]  # Limit to first few
    for idx, (c, s, theta) in enumerate(cs_theta_combinations[:4]):  # Show first 4
        if idx >= 4:
            break
        row = idx // 2
        col = idx % 2
        
        if (c, s, theta) in Ori_processing.O_dict and Ori_processing.O_dict[(c, s, theta)] is not None:
            result = Ori_processing.O_dict[(c, s, theta)]
            im = axes[row, col].imshow(cp.asnumpy(result), cmap='hot')
            axes[row, col].set_title(f'O_dict[{c},{s},{theta}°]\nShape: {result.shape}')
            axes[row, col].axis('off')
            plt.colorbar(im, ax=axes[row, col], fraction=0.046, pad=0.04)
        else:
            axes[row, col].text(0.5, 0.5, f'Missing:\n({c},{s},{theta}°)', 
                               horizontalalignment='center', verticalalignment='center',
                               transform=axes[row, col].transAxes, fontsize=12)
            axes[row, col].axis('off')
    
    # Hide unused subplots
    for idx in range(len(cs_theta_combinations), 4):
        if idx < 4:
            axes[idx // 2, idx % 2].axis('off')
    
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