import numpy as np
from utils import addition, subtraction
import cv2

def flicker_computation(Is0,Is1,args,ifshow = False)->list[np.array]:
    """
    compute the flicker map, which might be used to compute the saliency map.
    """
    flicker_maps = []
    for c in args.center:
        for s in args.surrounding:
            tmp_map = subtraction(Is0[c] - Is1[c], Is0[s] - Is1[s])
            flicker_maps.append(np.where(tmp_map > 0, tmp_map, 0))
            flicker_maps.append(np.where(tmp_map < 0, -tmp_map, 0))
    return flicker_maps
            

