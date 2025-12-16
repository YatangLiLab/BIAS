# BIAS: a Biologically Inspired Algorithm for video Saliency detection

This is the official code repo for the work **BIAS: a Biologically Inspired Algorithm for Video Saliency Detection**. The code for running the BIAS is in the src folder. The codes to evaluate or to plot the results are in the  eval and analysis folders, respectively. The path in each file might be altered to the real path on the user's platform.

---

A demo has been added to the repository. You may clone the repository and simply run the `demo.ipynb` in an environment that satisfies the requirements in the BIAS folder, and an example saliency video would be created in the root folder.

---

To convert only one RGB video into an `.mp4` saliency map, just run:

```bash
cd ./src
python ./main.py --video_path 'YOUR_VIDEO_PATH' --generate_name 'YOUR_TARGET_OUTPUT_PATH'
```
Since we use the `OpenCV` video stream to load images, it is convenient to use any camera as long as it can be loaded through `OpenCV` or turned into a video stream input.

If you want to process multiple videos, please use `src/batch.py`. The `images2images` mode is also available in `batch.py` and `main.py`; you may refer to the `argparse - help` code blocks for more information. 

We also have a code to process batches of images through `batch.py`. Currently, we only support manually changing the target directory and processing modes(image, motion, both, GWTA/MGF, or flicker). The default pool size is set to 4 in case of the memory overflow error on personal devices, achieving a 100 fps processing rate. If running on high-performance platforms, users might need to manually set a larger pool size. 

```bash
cd ./src
python ./batch.py
```

---

The training code, inference code, evaluation script, model configs, and some of the models are stored in the folder `MSTCN`. Currently, we do not provide the link to the dataset, but you may check and download it using `MSTCN/download_video.py` and convert it into `.npy` files for training.
