import statistics
from skimage import segmentation, color
import numpy as np
from skimage.color import rgb2hsv
import matplotlib.pyplot as plt
from skimage.segmentation import slic, mark_boundaries
import csv
import os
 
directory = "/Users/sunechristiansen/sune/ds_project/mdasm-2024/pictures/"
directory_mask = "/Users/sunechristiansen/sune/ds_project/mdasm-2024/groupR_masks/"
pic = os.listdir(directory)
pic_mask = os.listdir(directory_mask)
nameOfPic = []
H = []
S = []
V = []

for x in range(len(pic)):
    if os.path.exists(directory_mask+pic[x].split(".")[0]+'_mask'+".png"):
        rgb_img = plt.imread(directory+pic[x])[:,:,:3]
        mask = plt.imread(directory_mask+pic[x].split(".")[0]+'_mask'+".png")
        print((directory+pic[x]), (directory_mask+pic[x].split(".")[0]+'_mask'+".png"))
        img_avg_lesion = rgb_img.copy()
    else:
        continue
    nameOfPic.append(pic[x].split(".")[0])
    lesion_coords = np.where(mask != 0)
    min_x = min(lesion_coords[0])
    max_x = max(lesion_coords[0])
    min_y = min(lesion_coords[1])
    max_y = max(lesion_coords[1])
    cropped_lesion = rgb_img[min_x:max_x,min_y:max_y]

    segments = slic(cropped_lesion, n_segments=250, compactness=100)

    # Calculate the mean color for each segment
    segment_means = []
    for segment_value in np.unique(segments):
        mask = segments == segment_value
        segment_pixels = cropped_lesion[mask]
        segment_mean = segment_pixels.mean(axis=0)
        segment_means.append(segment_mean)

    sdt = segment_means

    sdt = np.array(sdt)

    sss = rgb2hsv(sdt)*100

    
    np.round(sss, decimals=0)

    h = []
    s = []
    v = []



    for x in sss:
        h.append(x[0].tolist())
        s.append(x[1].tolist())
        v.append(x[2].tolist())
        

    H.append(statistics.stdev(h)**2)
    S.append(statistics.stdev(s)**2)
    V.append(statistics.stdev(v)**2)




H.insert(0, 'H')
S.insert(0, 'S')
V.insert(0, 'V')
nameOfPic.insert(0, 'Name_Of_Pic')

rows = zip(nameOfPic, H, S, V, diagnostic)

with open('HSV_Var.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(rows)
