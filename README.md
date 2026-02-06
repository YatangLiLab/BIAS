# BIAS: a Biologically Inspired Algorithm for video Saliency detection

This is the official code repo for the work **BIAS: a Biologically Inspired Algorithm for Video Saliency Detection**. The code for running the **BIAS** on CPU is in the `BIAS` folder, and a version that could be executed on the Nvidia **GPU** platform through **Cupy** is provided in the `BIAS_GPU` folder. A [open-source CUDA-based convolution method](https://github.com/elcruzo/cuda-conv) is applied in our code.

The codes to evaluate or to plot the results we reported in the article are in the `eval` and `analysis` folders, respectively. The paths in each file should be altered to the real path on the user's platform.

The training codes of **Spark**&**MSTCN** are also provided in the `Spark` and `MSTCN` folders, and we provide some of their training config, prediction results, and evaluation metrics in them. Other codes for traffic anticipation, including [DSTA](https://github.com/monjurulkarim/DSTA), [DRIVE](https://github.com/Cogito2012/DRIVE), and [UString](https://github.com/Cogito2012/UString), are included and slightly modified in the  folder `TrafficAnticipationCodes`. These codes need to be executed in `CUDA 9.8-10.2` environments, due to their usage of [mmdetection v1.0.0](https://github.com/open-mmlab/mmdetection/releases/tag/v1.0.0), which only supports `Kepler, Maxwell, Pascal, Volta,` and `Turing` architecture GPU(We run these codes on a Turing architecture RTX2080Ti platform). The corresponding CUDA drivers are only designed for `Ubuntu 16.04` or `Ubuntu 18.04`. If you are using a later LTS `Ubuntu` system like `20.04`, this [installation guide](https://blog.csdn.net/mofy_/article/details/122791758) might help(though in Chinese).

To inference using the [Salfom](https://github.com/mr17m/SalFoM---Video-Saliency-Prediction) model, you may refer to [this repo](https://github.com/Zhang-Zhaoji/Full-SalFoM) with essential running files. We also updated the some of the weight-links  in the original repo.

---

A **demo** has been added to the repository. You may clone the repository and simply run the `demo.ipynb` in an environment that satisfies the requirements in the BIAS folder, and an example saliency video would be created in the root folder.

---

To convert only one RGB video into an `.mp4` saliency map(Using CPU edition), just run:

```bash
cd ./BIAS/
python ./main.py --video_path 'YOUR_VIDEO_PATH' --generate_name 'YOUR_TARGET_OUTPUT_PATH'
```

Since we use the `OpenCV` video stream to load images, it is convenient to use any camera as long as it can be loaded through `OpenCV` or turned into a video stream input.

If you want to process multiple videos, please use `src/batch.py`. The `images2images` mode is also available in `batch.py` and `main.py`; you may refer to the `argparse - help` code blocks for more information.

We also have a code to process batches of images through `batch.py`. Currently, we only support manually changing the target directory and processing modes(image, motion, both, GWTA/MGF, flicker or other methods). The default pool size is set to 4 in case of the memory overflow error on personal devices, achieving a 100 fps processing rate. If running on high-performance platforms, users might need to manually set a larger pool size.

```bash
cd ./BIAS/
python ./batch.py
```

The GPU edition works similarily.

---

The training code, inference code, evaluation script, model configs, and some of the models are stored in the folder `MSTCN`. Currently, we do not provide the link to the dataset, but you may check and download it using `MSTCN/download_video.py` and convert it into `.npy` files for training.

---

We provide the trained weights for the Spark-MSTCN models mentioned in the article through the Baidu Yun Netdisk. You may download them through the following link or click [this](https://pan.baidu.com/s/1kghhTqMRNG7gIwTXlkBAAg?pwd=1234). The available Cause-effect dataset is also provided in the folder online.

```
link: https://pan.baidu.com/s/1kghhTqMRNG7gIwTXlkBAAg?pwd=1234 
password: 1234 
```

---

If you want to cite our work, please use the following bibtex:

```
Just joking, we do not have a published paper yet.
```
