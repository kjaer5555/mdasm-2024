import matplotlib.pyplot as plt
import numpy as np
from skimage.color import rgb2hsv
from skimage.transform import resize
from skimage.io import imread

def detect_bwv(img, blue_mask):
    
    # Enhance blue channel
    factor = 2.5
    img_float = img.astype('float')
    # Scale blue channel
    img_float[:,:,2] = img_float[:,:,2] * factor
    # Clip values to max 255 (for uint8 images)
    img_float[:,:,2] = np.clip(img_float[:,:,2], 0, 255)
    # Convert back to original data type
    img = img_float.astype(img.dtype)

    # define bwv value
    values = {
        "hue_min": 0.55,
        "hue_max": 0.7,
        "sat_min": 0.0,
        "sat_max": 1.0,
        "val_min": 0.0,
        "val_max": 1.0
    }
    #get bwv value
    hsv_img = rgb2hsv(img)
    blue_mask = (hsv_img[..., 0] >= values["hue_min"]) & (hsv_img[..., 0] <= values["hue_max"]) & \
           (hsv_img[..., 1] >= values["sat_min"]) & (hsv_img[..., 1] <= values["sat_max"]) & \
           (hsv_img[..., 2] >= values["val_min"]) & (hsv_img[..., 2] <= values["val_max"])
    image_with_blue_mask = np.where(blue_mask[..., None], img, 0)
    
    #score
    treshold=0.05
    score = np.count_nonzero(image_with_blue_mask) / np.count_nonzero(img)
    bin_score=0
    if score >= treshold:
        bin_score=1
    
    return  bin_score

#some paths
img = plt.imread("pictures/...")
blue_mask = plt.imread("groupR_blue_masks/...")

bwv_score = detect_bwv(img,blue_mask)
#bwv_score now contains the value for the blue white veil feature