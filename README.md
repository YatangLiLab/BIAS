# BIAS: a Biologically Inspired Algorithm for video Saliency detection

This is the official code repo for the work **BIAS: a Biologically Inspired Algorithm for Video Saliency Detection**. The code for running the BIAS is in the src folder. The codes to evaluate or to plot the results are in the  eval and analysis folders, respectively. The path in each file might be altered to the real path on the user's platform.

---

To convert only one RGB video into an `.mp4` saliency map, just run:

```bash
cd ./src
python ./main.py --video_path 'YOUR_VIDEO_PATH' --generate_name 'YOUR_TARGET_OUTPUT_PATH'
```
Since we use the `OpenCV` video stream to load images, it is convenient to use any camera as long as it can be captured through `OpenCV`.

If you want to process multiple videos, please use `src/batch.py`. The `images2images` mode is also available in `batch.py` and `main.py`; you may refer to the `argparse - help` code blocks for more information. 

We also have a code to process batches of images through `batch.py`. Currently, we only support manually changing the target directory and processing modes(image, motion, both, GWTA/MGF, or flicker). The default pool size is set to 4 in case of the memory overflow error on personal devices, achieving a 100 fps processing rate. If running on high-performance platforms, users might need to manually set a larger pool size. 

```bash
cd ./src
python ./batch.py
```
