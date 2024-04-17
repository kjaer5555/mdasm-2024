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
from scipy.spatial.distance import cityblock

directory = "pictures/"
directory_mask = "groupR_masks/"
pictures = os.listdir(directory)
picturestures_mask = os.listdir(directory_mask)
nameOfPictures = []
hue_values = []
saturation_values = []
value_values = []

atypical_pigmentation_network = []

red_presence=[]
brown_presence=[]
blue_presence=[]
pink_presence=[]
black_presence=[]

blue_white_veil=[]

def is_darker(color1, color2):
    """ Input: Two combinations of rgb values
        Output: The result of the comparison of brightness between the two images. It searches the darker one."""
    # Convert colors to grayscale
    gray1 = 0.299 * color1[0] + 0.587 * color1[1] + 0.114 * color1[2]
    gray2 = 0.299 * color2[0] + 0.587 * color2[1] + 0.114 * color2[2]
    
    # Compare grayscale values
    return gray1 < gray2

def get_apn_score(cropped_lesion, cropped_lesion_mask ):
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

def color_extraction(segments_mean_in_hsv):
    colours_bins=[0,0,0,0,0,0,0,0,0,0,0]
    colours=[[355, 83, 93], #red(pantone) - 
         [5, 93, 100], #some other red - 
         [30, 100, 59], #brown - 
         [209, 24, 44], #black coral
         [210, 50, 80],  #some blue-gray
         [329,49,97], #persian pink - 
         [350,25,100], #pink - 
         [346,42,97], #sherbet pink - 
         [354,89,61], #ruby red - 
         [20,56,69]] #brown sugar - 
         #[24,49,24]] #bistre - dark brown - 
    multip=np.array([360,100,100])
    #finishing part of the rgb to hsv conversion
    segments_mean_in_hsv_scaled = segments_mean_in_hsv*multip
    for i in segments_mean_in_hsv_scaled:
        distance_color=[0,0,0,0,0,0,0,0,0,0]
        #if the v value of HSV is below 25, then it is black and goes to bin 5
        if int(i[2])<=25:
            colours_bins[10]+=1
        else:
            for j in enumerate(colours):
                #Calculate the Manhattans distance between the current pixel and the defined colours
                distance_color[j[0]]=cityblock(i, j[1])
            smallest_dist=min(distance_color)
            for k in enumerate(distance_color):
                if k[1]==smallest_dist:
                    colours_bins[k[0]]+=1
        #print(colours_bins)
        #thrashhold=0.05*
        #counter=0
    colour_bins_new=[colours_bins[0]+colours_bins[1]+colours_bins[8], colours_bins[2]+colours_bins[9], colours_bins[3]+colours_bins[4], colours_bins[5]+colours_bins[6]+colours_bins[7], colours_bins[10]]
    colour_bins_final=[a_bin/sum(colours_bins) for a_bin in colour_bins_new]
    return colour_bins_final
    
def color_variance(segments_mean_in_hsv):
        h = []
        s = []
        v = []
        #need comment
        for x in segments_mean_in_hsv:
            h.append(x[0].tolist())
            s.append(x[1].tolist())
            v.append(x[2].tolist())
            
        h = list(((np.array(h)*360 + 180)%360)/360)
        return h,s,v

def is_bwv(cropped_lesion):
    
    # Enhance blue channel
    factor = 2.5
    img_float = cropped_lesion.astype('float')
    # Scale blue channel
    img_float[:,:,2] = img_float[:,:,2] * factor
    # Clip values to max 255 (for uint8 images)
    img_float[:,:,2] = np.clip(img_float[:,:,2], 0, 255)
    # Convert back to original data type
    cropped_lesion = img_float.astype(cropped_lesion.dtype)

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
    hsv_img = rgb2hsv(cropped_lesion)
    blue_mask = (hsv_img[..., 0] >= values["hue_min"]) & (hsv_img[..., 0] <= values["hue_max"]) & \
           (hsv_img[..., 1] >= values["sat_min"]) & (hsv_img[..., 1] <= values["sat_max"]) & \
           (hsv_img[..., 2] >= values["val_min"]) & (hsv_img[..., 2] <= values["val_max"])
    image_with_blue_mask = np.where(blue_mask[..., None], cropped_lesion, 0)
    
    #score
    treshold=0.05
    score = np.count_nonzero(image_with_blue_mask) / np.count_nonzero(cropped_lesion)
    bin_score=0
    if score >= treshold:
        bin_score=1
    
    return  bin_score
     
 
i = 0
for x in range(len(pictures)):
    if os.path.exists(directory_mask+pictures[x].split(".")[0]+'_mask'+".png"):
        rgb_img = plt.imread(directory+pictures[x])[:,:,:3]
        mask = plt.imread(directory_mask+pictures[x].split(".")[0]+'_mask'+".png")
        #print((directory+pictures[x]), (directory_mask+pictures[x].split(".")[0]+'_mask'+".png"))
        print(i)
        i += 1

    else:
        continue
    nameOfPictures.append(pictures[x].split(".")[0])
    lesion_coords = np.where(mask != 0)
    min_x = min(lesion_coords[0])
    max_x = max(lesion_coords[0])
    min_y = min(lesion_coords[1])
    max_y = max(lesion_coords[1])
    cropped_lesion = rgb_img[min_x:max_x,min_y:max_y]
    cropped_lesion_mask=mask[min_x:max_x,min_y:max_y]

    segments = slic(cropped_lesion, n_segments=250, compactness=100)

    apn_score = get_apn_score(cropped_lesion, cropped_lesion_mask) 
    atypical_pigmentation_network.append(apn_score)

    # Calculate the mean color for each segment
    segment_means = []
    for segment_value in np.unique(segments):
        mask_of_segment = segments == segment_value
        segment_pixels = cropped_lesion[mask_of_segment]
        segment_mean = segment_pixels.mean(axis=0)
        segment_means.append(segment_mean)

    segments_mean_in_hsv = rgb2hsv(np.array(segment_means))

    colors_presence=color_extraction(segments_mean_in_hsv)
    red_presence.append(colors_presence[0])
    brown_presence.append(colors_presence[1])
    blue_presence.append(colors_presence[2])
    pink_presence.append(colors_presence[3])
    black_presence.append(colors_presence[4])
    
    h_pic,s_pic,v_pic=color_variance(segments_mean_in_hsv)
    hue_values.append(statistics.stdev(h_pic)**2)
    saturation_values.append(statistics.stdev(s_pic)**2)
    value_values.append(statistics.stdev(v_pic)**2)

    blue_white_veil.append(is_bwv(cropped_lesion))
    


df_features = pd.DataFrame()

df_features['Name_Of_Picture'] = nameOfPictures
df_features['H_value'] = hue_values
df_features['S_value'] = saturation_values
df_features['V_value'] = value_values
df_features['red_presence'] =red_presence
df_features['brown_presence'] =brown_presence
df_features['blue_presence'] =blue_presence
df_features['pink_presence'] =pink_presence
df_features['black_presence'] =black_presence
df_features['atypicturesal'] = atypical_pigmentation_network
df_features['blue-whiteveils'] = blue_white_veil



path_diagnostics_data = 'image_ids_groups.csv'

df_img_diadnostics = pd.read_csv(path_diagnostics_data)


# Assuming df_var and df_img are defined with appropriate data
diagnostic = []
cancer = []

# Create a dictionary from df_img for quick lookup
img_dict = {img_id: diag for img_id, diag in zip(df_img_diadnostics['img_id'], df_img_diadnostics['diagnostic'])}

for name in df_features['Name_Of_Picture']:
    img_name = name + '.png'
    if img_name in img_dict:
        diag = img_dict[img_name]
        diagnostic.append(diag)
        if diag in ['BCC', 'MEL', 'SCC']:
            cancer.append(1)
        else:
            cancer.append(0)
    else:
        # Handling cases where there is no match found
        diagnostic.append('Unknown')
        cancer.append(0)  # or np.nan to represent missing data if appropriate
        
df_features['diagnostic'] = diagnostic
df_features['cancer_or_not'] = cancer





df_features.to_csv('features.csv')
