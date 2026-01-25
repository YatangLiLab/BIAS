import numpy as np
import cupy as cp
import argparse
from cupyx.scipy.ndimage import convolve1d
from utils_cupy import CupyImageProcessing, cpnormalize_img

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
        return cp.abs(self.image_processing.subtraction_torch(self.Os[theta][c], self.Os[theta][s]))
    
    def Orientation_maps(self,ifshow=False):
        for c,s in self.cs_lst:
            for theta in self.theta_set:
                self.O_dict[(c,s,theta)] = self.O_c_s_theta(c,s,theta//45)
        if ifshow:
            raise NotImplementedError('not implement if show methods now')
    
    def synthesis_O_map(self, ifshow=False):
        O_bar_0 = cp.zeros((1,1))
        O_bar_45 = cp.zeros((1,1))
        O_bar_90 = cp.zeros((1,1))
        O_bar_135 = cp.zeros((1,1))
        for c,s in self.cs_lst:
            O_bar_0 = self.image_processing.addition_torch(O_bar_0, cpnormalize_img(self.O_dict[c,s,0]))
            O_bar_45 = self.image_processing.addition_torch(O_bar_45, cpnormalize_img(self.O_dict[c,s,45]))
            O_bar_90 = self.image_processing.addition_torch(O_bar_90, cpnormalize_img(self.O_dict[c,s,90]))
            O_bar_135 = self.image_processing.addition_torch(O_bar_135, cpnormalize_img(self.O_dict[c,s,135]))
        O_bar_a = self.image_processing.addition_torch(cpnormalize_img(O_bar_0), cpnormalize_img(O_bar_45))
        O_bar_b = self.image_processing.addition_torch(cpnormalize_img(O_bar_90), cpnormalize_img(O_bar_135))
        return self.image_processing.addition_torch(O_bar_a, O_bar_b)
    
    def build_pyramid(self, Is:list[cp.uint8]):
        for idx in range(1,self.pyramid_height):
            if idx not in self.cal_lst:
                for _ in range(len(self.theta_set)):
                    self.Os[_][idx] = (cp.zeros((1,1)))
            else:
                for _ in range(len(self.theta_set)):
                    self.Os[_][idx] = (self.gabor_filters[0][_](Is[idx]))
        for idx in range(self.pyramid_height,self.total_height):
            if idx not in self.cal_lst:
                for _ in range(len(self.theta_set)):
                    self.Os[_][idx] = (cp.zeros((1,1)))
            else:
                for _ in range(len(self.theta_set)):
                    self.Os[_][idx] = (self.gabor_filters[idx-self.pyramid_height+1][_](Is[idx]))


if __name__ == '__main__':
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
        parse.add_argument('--center',default = (2,3,4),type=tuple,help = "center params. Itti default params are (2, 3, 4)")
        parse.add_argument('--surrounding',default = (3,4),type=tuple,help = "surrounding params. Itti default params are (3, 4)")
        args = parse.parse_args() 
        return args

    args = parse_args()