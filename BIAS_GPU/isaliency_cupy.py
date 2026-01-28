import numpy as np
import cupy as cp
import argparse
import cv2
from cupyx.scipy.ndimage import zoom, convolve1d
from utils_cupy import cpnormalize_img, cpnormalize_img3d_dict, CupyImageProcessing, CupyImagePyramid


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
    def __init__(self, args, shape_hw):
        self.inv_gamma = 1.0 / args.gamma_correction
        H, W = shape_hw
        
        # 预计算 Mask 并直接存入 GPU (建议用 float32 保证 Gamma 精度，后续再转 float16)
        k_h = cp.asarray(cv2.getGaussianKernel(H, H / 2.0))
        k_w = cp.asarray(cv2.getGaussianKernel(W, W / 2.0))
        self.mask_gpu = (k_h @ k_w.T).astype(cp.float16)
        self.mask_gpu = self.mask_gpu / cp.max(self.mask_gpu)
        self.neg_mask_gpu = 1.0 - self.mask_gpu

    def process(self, image_cpu):
        # 1. 将输入拷贝到 GPU
        img_gpu = cp.asarray(image_cpu, dtype=cp.float16)
        img_norm = img_gpu / 255.0
        img_mean = cp.mean(img_norm)
        
        return self._fused_process(
            img_norm, 
            self.mask_gpu[..., cp.newaxis], 
            self.neg_mask_gpu[..., cp.newaxis], 
            self.inv_gamma, 
            img_mean
        ).astype(cp.float16)

    @cp.fuse()
    def _fused_process(img_norm, mask, neg_mask, inv_gamma, m_val):
        corrected = cp.power(img_norm, inv_gamma)
        return (mask * corrected + neg_mask * m_val) * 255.0


class ICpyrimid:
    _fast_build_kernel = cp.ElementwiseKernel(
        'T R, T G, T B, T max_val', # 输入
        'raw T out',                # 输出
        '''
        // 1. 使用 'i' 代替 '_ind' 获取线性索引
        long long base = i * 6;

        T s_val = (R + G + B) * (T)0.5;
        T i_val = (T)0.299 * R + (T)0.587 * G + (T)0.114 * B;

        // 2. 计算各个分量，预先算好
        T v1 = R * (T)1.5 - s_val;
        T v2 = B * (T)1.5 - s_val;
        T v3 = max_val - i_val;
        T v4 = G * (T)1.5 - s_val;
        
        T diff = G - R;
        T abs_diff = (diff < (T)0.0) ? (T)(-diff) : diff;
        T v5 = (G + R) * (T)0.5 - abs_diff * (T)0.5 - B;

        // 3. 写入显存，使用三元运算符代替 max 以兼容 float16
        out[base + 0] = i_val;
        out[base + 1] = (v1 > (T)0.0) ? v1 : (T)0.0;
        out[base + 2] = (v2 > (T)0.0) ? v2 : (T)0.0;
        out[base + 3] = (v3 > (T)0.0) ? v3 : (T)0.0;
        out[base + 4] = (v4 > (T)0.0) ? v4 : (T)0.0;
        out[base + 5] = (v5 > (T)0.0) ? v5 : (T)0.0;
        ''',
        'fast_build_6_channels'
    )

    def __init__(self, args):
        self.args = args
        self.c_set = sorted(self.args.center)
        self.delta_set = sorted(self.args.surrounding)
        self.cs_lst = [(c,c+d) for c in self.c_set for d in self.delta_set]
        self.cal_lst = list(set([c for (c,s) in self.cs_lst] + [s for (c,s) in self.cs_lst]))
        self.total_height = args.total_height
        self.build_pyramid = CupyImagePyramid()
        self.img_process = CupyImageProcessing()
        self.transformI = cp.asarray([0.299, 0.587, 0.114]).astype(cp.float16)
        self.shapes = [(240, 320), (120, 160), (60, 80), (30, 40), (15, 20), (8, 10), (4, 5), (2, 3)]
        self.final_channel_cnt = 6
        self.reset()

    def reset(self):
        self.ICs = [cp.zeros(shape + (self.final_channel_cnt,), dtype=cp.float16) for shape in self.shapes]
        self.scaling_dict = {(c,s):cp.zeros_like(self.ICs[c], dtype=cp.float16) for (c,s) in self.cs_lst}
    

    def build(self, gaussian_img):
        self.reset()
        # 1. 强制转换输入为 float16
        img = gaussian_img.astype(cp.float16, copy=False)
        R, G, B = img[..., 2], img[..., 1], img[..., 0]
        
        # 2. 计算全局最大值并确保也是 float16 标量
        max_val = cp.max(img).astype(cp.float16)
        
        # 3. 确保输出容器是连续的且类型正确
        if self.ICs[0].dtype != cp.float16:
            self.ICs[0] = self.ICs[0].astype(cp.float16)
        
        # 显存连续性检查（重要）
        if not self.ICs[0].flags.c_contiguous:
            self.ICs[0] = cp.ascontiguousarray(self.ICs[0])

        # 4. 调用内核
        self._fast_build_kernel(R, G, B, max_val, self.ICs[0])

        # 5. 构建后续金字塔
        self.build_pyramid.eight_pyramid_built_3d(self.ICs)
    
    def scaling(self, c,s):
        diff = self.ICs[c][:,:,:3] - self.ICs[c][:,:,3:]
        diff = self.img_process.subtraction_torch(diff, self.ICs[s][:,:,:3] - self.ICs[s][:,:,3:])
        diff -= cp.mean(diff, axis=(0,1), keepdims=True)
        self.scaling_dict[(c,s)] = cp.concatenate((cp.maximum(diff, 0), cp.maximum(-diff, 0)), axis=2).astype(cp.float16)
        

    def diff_process(self):
        for (c,s) in self.cs_lst:
            self.scaling(c,s)
        cpnormalize_img3d_dict(self.scaling_dict)
    
    def get_conspicuous_map(self):
        IC_bar = cp.zeros((1,1,3))
        for (c,s) in self.cs_lst:
            IC_bar = self.img_process.addition_torch(IC_bar, self.scaling_dict[(c,s)][:,:,:3] + self.scaling_dict[(c,s)][:,:,3:])
        I_bar = IC_bar[..., 0]
        C_bar = cp.sum(IC_bar[..., 1:], axis=-1)
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
        parse.add_argument('--default_size',default=(320,240),type=tuple,help = 'default size of image')
        parse.add_argument('--center',default = (1,2),type=tuple,help = "center params. Itti default params are (2, 3, 4)")
        parse.add_argument('--surrounding',default = (3,4),type=tuple,help = "surrounding params. Itti default params are (3, 4)")
        parse.add_argument('--gamma_correction', type=float, default=2.2, help='Gamma correction value')
        args = parse.parse_args() 
        return args
    
    args = parse_args()
    
    # Load the test image from the provided path
    test_image_path = '/mnt/e/Li Lab/CVPR_rebuttal_/cuda-conv/data/lena.png'
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
