import statistics
from skimage import segmentation, color
import numpy as np
from skimage.color import rgb2hsv
import matplotlib.pyplot as plt
from skimage.segmentation import slic, mark_boundaries
import csv
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
 
directory = "/Users/sunechristiansen/sune/ds_project/mdasm-2024/pictures/"
directory_mask = "/Users/sunechristiansen/sune/ds_project/mdasm-2024/groupR_masks/"
pic = os.listdir(directory)
pic_mask = os.listdir(directory_mask)
nameOfPic = []
H = []
S = []
V = []

atypical = []

def is_darker(color1, color2):
    """ Input: Two combinations of rgb values
        Output: The result of the comparison of brightness between the two images. It searches the darker one."""
    # Convert colors to grayscale
    gray1 = 0.299 * color1[0] + 0.587 * color1[1] + 0.114 * color1[2]
    gray2 = 0.299 * color2[0] + 0.587 * color2[1] + 0.114 * color2[2]
    
    # Compare grayscale values
    return gray1 < gray2

def get_apn(cropped_lesion, cropped_lesion_mask ):
    """ Input: Croped version of the image and its mask that contains the major area of the lesion. 
        Output: The proportion of overlapp between the darker segmented area and the cropped original mask. """
    #flatten the image
    reshape=cropped_lesion.reshape(-1,3)
    reshape.shape

    #cluster all pixels into 2 segments
    k_clust=KMeans(n_clusters=2, n_init=10)
    k_clust.fit(reshape)
    segmentation_labels=k_clust.labels_ #gets the name of the 2 segments
    centroids = k_clust.cluster_centers_ #gets the colour value of the centeroid for the two clusters

    label_image = segmentation_labels.reshape(cropped_lesion_mask.shape[:2])#reshapes the segmented images so that they can be compared to the cropped mask

    cluster_image_1 = np.zeros_like(cropped_lesion_mask) #segment 1
    cluster_image_2 = np.zeros_like(cropped_lesion_mask) #segment 2

    cluster_image_1[label_image == 0] = cropped_lesion_mask[label_image == 0] #mask of segment 1
    cluster_image_2[label_image == 1] = cropped_lesion_mask[label_image == 1] #mask of segment 2

        
    if is_darker(tuple(centroids[0]), tuple(centroids[1])): # find the darker segment
        major_cl=cluster_image_1
        overlapping_area = np.sum(cropped_lesion_mask * major_cl)

    else:
        major_cl=cluster_image_2
        overlapping_area = np.sum(cropped_lesion_mask * major_cl)

    return overlapping_area/ np.sum(cropped_lesion_mask) #calculate the overlap between the darker segment and the original mask


i = 0
for x in range(len(pic)):
    if os.path.exists(directory_mask+pic[x].split(".")[0]+'_mask'+".png"):
        rgb_img = plt.imread(directory+pic[x])[:,:,:3]
        mask = plt.imread(directory_mask+pic[x].split(".")[0]+'_mask'+".png")
        #print((directory+pic[x]), (directory_mask+pic[x].split(".")[0]+'_mask'+".png"))
        print(i)
        i += 1
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
    cropped_lesion_mask=mask[min_x:max_x,min_y:max_y]

    segments = slic(cropped_lesion, n_segments=250, compactness=100)

    aty = get_apn(cropped_lesion, cropped_lesion_mask)

    atypical.append(aty)

    # Calculate the mean color for each segment
    segment_means = []
    for segment_value in np.unique(segments):
        mask = segments == segment_value
        segment_pixels = cropped_lesion[mask]
        segment_mean = segment_pixels.mean(axis=0)
        segment_means.append(segment_mean)

    sdt = segment_means

    sdt = np.array(sdt)

    sss = rgb2hsv(sdt)

    

    
    np.round(sss, decimals=0)

    h = []
    s = []
    v = []
    #need comment
    h = list(((np.array(h)*360 + 180)%360)/360)
    

    for x in sss:
        h.append(x[0].tolist())
        s.append(x[1].tolist())
        v.append(x[2].tolist())
        

    H.append(statistics.stdev(h)**2)
    S.append(statistics.stdev(s)**2)
    V.append(statistics.stdev(v)**2)



df_var = pd.DataFrame()

df_var['Name_Of_Pic'] = nameOfPic
df_var['H'] = H
df_var['S'] = S
df_var['V'] = V
df_var['atypical'] = atypical



path_img = 'image_ids_groups.csv'

df_img = pd.read_csv(path_img)


i = 0
j = 0

diagnostic = []
cancerOrNot = []



while j <= (len(df_var['Name_Of_Pic'])-1):
    i = 0
    for x in range(len(df_img['img_id'])):
        if df_img['img_id'][i] == df_var['Name_Of_Pic'][j]+'.png':
            #print(df_img['diagnostic'][j], df_img['img_id'][i])
            diagnostic.append(df_img['diagnostic'][j])
            if df_img['diagnostic'][j] == 'BCC' or df_img['diagnostic'][j] == 'MEL' or df_img['diagnostic'][j] == 'SCC':
                cancerOrNot.append(1)
            else:
                cancerOrNot.append(0)
            i += 1
            j += 1
        else:
            i += 1
        
df_var['diagnostic'] = diagnostic
df_var['cancer_or_not'] = cancerOrNot





df_var.to_csv('features1.csv')
