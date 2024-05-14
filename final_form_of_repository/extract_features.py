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
from skimage.transform import resize, rotate
import math

# Load image directories
#directory = "images/"
#directory_mask = "masks/"
#pictures = os.listdir(directory)

#Feature extraction functions
def is_darker(color1, color2):
    """ Input: Two combinations of rgb values
        Output: The result of the comparison of brightness between the two images. It searches the darker one."""
    # Convert colors to grayscale
    gray1 = 0.299 * color1[0] + 0.587 * color1[1] + 0.114 * color1[2]
    gray2 = 0.299 * color2[0] + 0.587 * color2[1] + 0.114 * color2[2]
    
    # Compare grayscale values
    return gray1 < gray2

def asymmetry_score_fully_rotated(img, n_steps: int = 10):
    """Rotates images step times around axis, and computes overlapping percentage
    Best score: 1, worst score: 0. Insert cropped image"""
    if (n_steps <  1 or n_steps > 180):
        raise Exception('Amount of steps of range [1, 180]')

    #img = get_cropped(img)
    sum_of_overlapping_areas = 0
    total_area = np.sum(img)*n_steps
    step_size = math.ceil(180 / n_steps)
    
    for angle in range(0, 180, step_size):
        
        rotated_mask = rotate(img, angle, resize=True)
        # There is no need to crop the image to boundary boxes, 
        # because the center remains in the middle.
        middle_column = rotated_mask.shape[1] // 2
        left_half = rotated_mask[:, :middle_column]
        flipped_right_half = np.fliplr(rotated_mask[:, -middle_column:])
        
        overlap = left_half * flipped_right_half
        
        overlapping_area = np.sum(overlap)
        sum_of_overlapping_areas += overlapping_area
        
    score = (2*sum_of_overlapping_areas / total_area) # need to account for both halves
    return score

def asymmetry_classic(cropped_lesion_mask):
    """ Input: Cropped version of the lesion mask.
        Method: Finds major axis using minimum bounding box methods from multiple angle rotations. 
        Calculates overlap percentage acording to the minimal axis.
        Output: Asymetry score of the lesion between 0 (no symetry) and 1 (full symety). """

    min_bounding_box = cropped_lesion_mask
    min_bounding_box_size = np.size(cropped_lesion_mask)
    i=0
    total_area = np.sum(cropped_lesion_mask)
    #rotate the image and mesure 
    for angle in range(0, 180, 1):
        
        rotated_mask = rotate(cropped_lesion_mask, angle, resize=True)
        
        bounding_box_size = np.size(rotated_mask)
        
        i+=1
        if(min_bounding_box_size > bounding_box_size):
            min_bounding_box_size = bounding_box_size
            min_bounding_box = rotated_mask
           
    if (min_bounding_box.shape[0] > min_bounding_box.shape[1]):
    # Use vertical axis as symmetry axis
        middle_column = min_bounding_box.shape[1] // 2
        left_half = min_bounding_box[:, :middle_column]
        flipped_right_half = np.fliplr(min_bounding_box[:, -middle_column:])
        overlap = left_half * flipped_right_half
    else:
    # Use horizontal axis as symmetry axis
        middle_row = min_bounding_box.shape[0] // 2  
        top_half = min_bounding_box[:middle_row, :]
        flipped_bottom_half = np.flipud(min_bounding_box[-middle_row:, :])  
        overlap = top_half * flipped_bottom_half 

    return 2* np.sum(overlap)/ total_area

def get_apn_score(cropped_lesion, cropped_lesion_mask ):
    """ Input: Cropped version of the image and its mask that contains the major area of the lesion. 
        Method: Using K-means clustering we determine two segments in the cropped section (a dark one and a bright one). 
        The dark one is considered the primary source of the disease.
        Output: The proportion of overlapp between the darker segmented area and the cropped original mask. """
    #flatten the image
    reshape=cropped_lesion.reshape(-1,3)
    reshape.shape

    #cluster all pixels into 2 segments
    k_clust=KMeans(n_clusters=2, n_init=10)
    k_clust.fit(reshape)
    segmentation_labels=k_clust.labels_ #gets the name of the 2 segments
    centroids = k_clust.cluster_centers_ #gets the color value of the centeroid for the two clusters

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
    """ Input: Mean HSV values of each megapixel in the cropped segment of the original picture.
        Method: Calculates the Manhattan distance of each mean value to the predefined color categories 
        and takes the minimal found. All segments are distibuted among the major color labels 
        and their relative presence in the lesion is calculated. 
        Output: Relative presence of each major color found in lesions
    """
    colors_bins=[0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    colors=[[355, 83, 93], #red(pantone) - ready
         [5, 93, 100], #some other red - ready
         [30, 100, 59], #brown - ready
         [209, 24, 44], #black coral
         [210, 50, 80],  #some blue-gray
         [329,49,97], #persian pink - ready
         [350,25,100], #pink - ready
         [346,42,97], #sherbet pink - ready
         [354,89,61], #ruby red - ready
         [20,56,69], #brown sugar - ready
         [24,49,24], #bistre - dark brown - ready
         [50,48,99], # yellow(crayola)
         [0,2,76]] #white
    multip=np.array([360,100,100])
    #finishing part of the rgb to hsv conversion
    segments_mean_in_hsv_scaled = segments_mean_in_hsv*multip
    for i in segments_mean_in_hsv_scaled:
        distance_color=[0,0,0,0,0,0,0,0,0,0,0,0,0]
        #if the v value of HSV is below 25, then it is black and goes to bin 5
        if int(i[2])<=25:
            colors_bins[10]+=1
        else:
            for j in enumerate(colors):
                #Calculate the Manhattans distance between the current pixel and the defined colors
                distance_color[j[0]]=cityblock(i, j[1])
            smallest_dist=min(distance_color)
            for k in enumerate(distance_color):
                if k[1]==smallest_dist:
                    colors_bins[k[0]]+=1
      
    color_bins_new=[colors_bins[0]+colors_bins[1]+colors_bins[8], colors_bins[2]+colors_bins[9]+colors_bins[10],
                 colors_bins[3]+colors_bins[4], colors_bins[5]+colors_bins[6]+colors_bins[7], 
                 colors_bins[13], colors_bins[11]+colors_bins[12]] #color reassignment
    color_bins_final=[a_bin/sum(colors_bins) for a_bin in color_bins_new] #relative presence
    return color_bins_final
    
def color_variance(segments_mean_in_hsv):
    """ Input: Mean HSV values of each megapixel in the cropped segment of the original picture.
        Method: Calculates standard deviation between the mean values extracted from the segments for each channel. 
        Output: Standard deviation of the HSV values in the cropped area.
    """
    h = []
    s = []
    v = []
    
    for x in segments_mean_in_hsv: #allocating values to the proper channel
        h.append(x[0].tolist())
        s.append(x[1].tolist())
        v.append(x[2].tolist())

    h = list(((np.array(h)*360 + 180)%360)/360) # solution to the misrepresentation of hsv scale

    return statistics.stdev(h)**2,statistics.stdev(s)**2,statistics.stdev(v)**2

def is_bwv(cropped_lesion):
    """ Input: Cropped version of the image.
        Method: Apply a predefined blue-white filter to the image and measure relative presence. 
        Output: Presence of blue-white veil in the lesion.
    """
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
    #build bwv value filter
    hsv_img = rgb2hsv(cropped_lesion)
    blue_mask = (hsv_img[..., 0] >= values["hue_min"]) & (hsv_img[..., 0] <= values["hue_max"]) & \
           (hsv_img[..., 1] >= values["sat_min"]) & (hsv_img[..., 1] <= values["sat_max"]) & \
           (hsv_img[..., 2] >= values["val_min"]) & (hsv_img[..., 2] <= values["val_max"])
    image_with_blue_mask = np.where(blue_mask[..., None], cropped_lesion, 0)
    
    #score of relative presence
    treshold=0.05
    score = np.count_nonzero(image_with_blue_mask) / np.count_nonzero(cropped_lesion)
    bin_score=0
    if score >= treshold:
        bin_score=1
    
    return  bin_score
     

def extract_features(imagepath,maskpath):
    """ Input: Path of the image, path of the mask.
        Method: Apply handmade mask to the image and extract each required feature.
        Output: Dataframe of the scores for the features.
    """
    rgb_img = plt.imread(imagepath)[:,:,:3]
    mask = plt.imread(maskpath)

    df_features = pd.DataFrame() #data frame to be returned

    #crop out lesion area
    lesion_coords = np.where(mask != 0)
    min_x = min(lesion_coords[0])
    max_x = max(lesion_coords[0])
    min_y = min(lesion_coords[1])
    max_y = max(lesion_coords[1])
    cropped_lesion = rgb_img[min_x:max_x,min_y:max_y] #cropped section from the original picture
    cropped_lesion_mask=mask[min_x:max_x,min_y:max_y] #cropped section from the original mask
    size=max(max_x-min_x, max_y-min_y) #find the longest height/lenght in pixels
    multiplier=250/size #create a multiplier which would make the longest distance equal to 250 pixels
    cropped_lesion=resize(cropped_lesion, (cropped_lesion.shape[0] * multiplier, cropped_lesion.shape[1] * multiplier), anti_aliasing=False)
    cropped_lesion_mask=resize(cropped_lesion_mask, (cropped_lesion_mask.shape[0] * multiplier, cropped_lesion_mask.shape[1] * multiplier), anti_aliasing=False)
    
    # Calculate the mean color for each segment
    segments = slic(cropped_lesion, n_segments=250, compactness=100) #devide image into megapixels
    segment_means = []
    for segment_value in np.unique(segments): #calculates the mean color values for each segment in found
        mask_of_segment = segments == segment_value
        segment_pixels = cropped_lesion[mask_of_segment]
        segment_mean = segment_pixels.mean(axis=0)
        segment_means.append(segment_mean)

    segments_mean_in_hsv = rgb2hsv(np.array(segment_means))
    colors_presence=color_extraction(segments_mean_in_hsv)
    h_pic,s_pic,v_pic=color_variance(segments_mean_in_hsv)
    try:
        imagename=imagepath.split('/')[-1].split('.')[0]
    except:
        imagename=imagepath.split('.')[0]
    df_features['Name_Of_Picture'] = [imagename]
    df_features['asymmetry_values'] = [asymmetry_score_fully_rotated(cropped_lesion_mask)]
    df_features['H_value'] = [h_pic]
    df_features['S_value'] = [s_pic]
    df_features['V_value'] = [v_pic]
    df_features['red_presence'] =[colors_presence[0]]
    df_features['brown_presence'] =[colors_presence[1]]
    df_features['blue_presence'] =[colors_presence[2]]
    df_features['pink_presence'] =[colors_presence[3]]
    df_features['black_presence'] =[colors_presence[4]]
    df_features['white_presence'] =[colors_presence[5]]
    df_features['atypical_pigment_network'] = [get_apn_score(cropped_lesion, cropped_lesion_mask)]
    df_features['blue-white_veil'] = [is_bwv(cropped_lesion)]
    
    return df_features

'''
df = pd.DataFrame(columns=['Name_Of_Picture','asymmetry_values','H_value','S_value','V_value','red_presence','brown_presence','blue_presence','pink_presence','black_presence','white_presence','atypical_pigment_network','blue-white_veil'])
n=len(pictures)
#n=40
print("extracting features from {} pictures".format(n))
for x in range(n): #loop through all pictures in the database and extract their features
    row=extract_features(pictures[x])
    if len(row)!=0:
        df=pd.concat([df,row])
    print(x)
print(df)
'''