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

# Load image directories
directory = "pictures/"
directory_mask = "groupR_masks/"
pictures = os.listdir(directory)
picturestures_mask = os.listdir(directory_mask)

#Prepare data holders
nameOfPictures = []
hue_values = []
saturation_values = []
value_values = []
asymmetry_values = []

atypical_pigment_network = []

red_presence=[]
brown_presence=[]
blue_presence=[]
pink_presence=[]
black_presence=[]
white_presence=[]

blue_white_veil=[]

#Feature extraction functions
def is_darker(color1, color2):
    """ Input: Two combinations of rgb values
        Output: The result of the comparison of brightness between the two images. It searches the darker one."""
    # Convert colors to grayscale
    gray1 = 0.299 * color1[0] + 0.587 * color1[1] + 0.114 * color1[2]
    gray2 = 0.299 * color2[0] + 0.587 * color2[1] + 0.114 * color2[2]
    
    # Compare grayscale values
    return gray1 < gray2

def get_cropped(img, padding=0):
    """ Input: Original image
        Output: Reduced to bounding boxes version of the original image with given padding."""
    # Find coordinates of non-zero pixels
    cords = np.where(img != 0)
    # Determine the bounds of the non-zero pixels
    x_min, x_max = np.min(cords[0]), np.max(cords[0])
    y_min, y_max = np.min(cords[1]), np.max(cords[1])
    # Apply padding, ensuring we do not go out of image bounds
    x_min = max(x_min - padding, 0)
    x_max = min(x_max + padding, img.shape[0] - 1)
    y_min = max(y_min - padding, 0)
    y_max = min(y_max + padding, img.shape[1] - 1)
    # Crop the image
    cropped_img = img[x_min:x_max+1, y_min:y_max+1]
    return cropped_img

# PLEASE try other version of the asymetry from the asymmetry.ipynb, there are 4 at least. Check which yields
# highest predicitve value.
def asymmetry_classic(img):
    """ Finds major axis using minimum bounding box methods and return combined overlap percentage"""
    initial_bounding_box = get_cropped(img, padding=0)
    # Lexographic order

    
    min_bounding_box = initial_bounding_box
    min_bounding_box_size = np.size(initial_bounding_box)
    i=0
    total_area = np.sum(initial_bounding_box)
    #print(min_bounding_box_size)
    
    for angle in range(0, 180, 1):
        
        rotated_mask = rotate(initial_bounding_box, angle, resize=True)
        bounding_box = get_cropped(rotated_mask, padding=0)
        bounding_box_size = np.size(bounding_box)
        #print(f'step {i} size {bounding_box_size} angle {angle}')
        i+=1
        if(min_bounding_box_size > bounding_box_size):
            min_bounding_box_size = bounding_box_size
            min_bounding_box = bounding_box
            #print(f'hit')
    
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
    
    overlapping_area = np.sum(overlap)
    score = (2*overlapping_area / total_area)
    
    return score

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
    colors=[[355, 83, 93], #red(pantone)
         [5, 93, 100], #some other red
         [30, 100, 59], #brown
         [209, 24, 44], #black coral
         [210, 50, 80],  #blue-gray(crayola)
         [329,49,97], #persian pink
         [350,25,100], #pink
         [346,42,97], #sherbet pink
         [354,89,61], #ruby red
         [20,56,69], #brown sugar
         [24,49,24], #bistre - dark brown
         [50,48,99], # yellow(crayola)
         [0,2,76]] #snow
    multip=np.array([360,100,100])
    #finishing part of the rgb to hsv conversion
    segments_mean_in_hsv_scaled = segments_mean_in_hsv*multip
    for i in segments_mean_in_hsv_scaled:
        distance_color=[0,0,0,0,0,0,0,0,0,0,0,0,0]
        #if the V value of HSV is below 25, then it is black and goes to bin 13
        if int(i[2])<=25:
            colors_bins[13]+=1
        else:
            for j in enumerate(colors):
                #Calculate the Manhattan's distance between the current pixel and the defined colors
                distance_color[j[0]]=cityblock(i, j[1])
            #Find which of the calculated distances is the shortest
            smallest_dist=min(distance_color)
            #Increase the number of pixels in the corresponding bin by 1
            for k in enumerate(distance_color):
                if k[1]==smallest_dist:
                    colors_bins[k[0]]+=1
    #rearrangement rearranging the different shades to the corresponding bin color
    color_bins_new=[colors_bins[0]+colors_bins[1]+colors_bins[8], #red
                 colors_bins[2]+colors_bins[9]+colors_bins[10], #brown
                 colors_bins[3]+colors_bins[4], #blue
                 colors_bins[5]+colors_bins[6]+colors_bins[7], #pink
                 colors_bins[13], #black
                 colors_bins[11]+colors_bins[12]] #white
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
     
 
i = 1
for x in range(len(pictures)): #loop through all pictures in the database and extract their features
    if os.path.exists(directory_mask+pictures[x].split(".")[0]+'_mask'+".png"):
        rgb_img = plt.imread(directory+pictures[x])[:,:,:3]
        mask = plt.imread(directory_mask+pictures[x].split(".")[0]+'_mask'+".png")
        
        print(i)
        i += 1

    else:
        continue

    #crop out lesion area
    nameOfPictures.append(pictures[x].split(".")[0])
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
    apn_score = get_apn_score(cropped_lesion, cropped_lesion_mask) 
    atypical_pigment_network.append(apn_score)
    asymmetry_values.append(asymmetry_classic(cropped_lesion_mask))
    blue_white_veil.append(is_bwv(cropped_lesion))

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
    red_presence.append(colors_presence[0])
    brown_presence.append(colors_presence[1])
    blue_presence.append(colors_presence[2])
    pink_presence.append(colors_presence[3])
    black_presence.append(colors_presence[4])
    white_presence.append(colors_presence[5])
    h_pic,s_pic,v_pic=color_variance(segments_mean_in_hsv)
    hue_values.append(h_pic)
    saturation_values.append(s_pic)
    value_values.append(v_pic)


#Build features database
df_features = pd.DataFrame()

df_features['Name_Of_Picture'] = nameOfPictures
df_features['asymmetry_values'] = asymmetry_values
df_features['H_value'] = hue_values
df_features['S_value'] = saturation_values
df_features['V_value'] = value_values
df_features['red_presence'] =red_presence
df_features['brown_presence'] =brown_presence
df_features['blue_presence'] =blue_presence
df_features['pink_presence'] =pink_presence
df_features['black_presence'] =black_presence
df_features['white_presence'] =white_presence
df_features['atypical_pigment_network'] = atypical_pigment_network
df_features['blue-white_veil'] = blue_white_veil



#get original diagnostics
path_diagnostics_data = 'image_ids_groups.csv'

df_img_diadnostics = pd.read_csv(path_diagnostics_data)

diagnostic = []
cancer = []

# Create a dictionary from df_img for quick lookup
img_dict = {img_id: diag for img_id, diag in zip(df_img_diadnostics['img_id'], df_img_diadnostics['diagnostic'])}

for name in df_features['Name_Of_Picture']:
    img_name = name + '.png'
    if img_name in img_dict:
        diag = img_dict[img_name]
        diagnostic.append(diag)
        if diag in ['BCC', 'MEL', 'SCC']: #cancerous diagnoses
            cancer.append(1)
        else:
            cancer.append(0)
    else:
        # Handling cases where there is no match found
        diagnostic.append('Unknown')
        cancer.append(0)  # or np.nan to represent missing data if appropriate
        
df_features['diagnostic'] = diagnostic
df_features['cancer_or_not'] = cancer

#export all data into a file
df_features.to_csv('features.csv')
