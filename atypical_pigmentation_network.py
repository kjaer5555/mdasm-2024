
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np

#read image
rgb_img = plt.imread("mdasm-2024\pictures\PAT_682_1292_642.png")[:,:,:3]
mask = plt.imread("mdasm-2024\groupR_masks\PAT_682_1292_642_mask.png")

def is_darker(color1, color2):
    """ Input: Two combinations of rgb values
        Output: The result of the comparison of brightness between the two images. It searches the darker one."""
    # Convert colors to grayscale
    gray1 = 0.299 * color1[0] + 0.587 * color1[1] + 0.114 * color1[2]
    gray2 = 0.299 * color2[0] + 0.587 * color2[1] + 0.114 * color2[2]
    
    # Compare grayscale values
    return gray1 < gray2

#crop the picture and the mask so that we can analyse the maj
lesion_coords = np.where(mask != 0)
min_x = min(lesion_coords[0])
max_x = max(lesion_coords[0])
min_y = min(lesion_coords[1])
max_y = max(lesion_coords[1])
cropped_lesion = rgb_img[min_x:max_x,min_y:max_y]
cropped_lesion_mask=mask[min_x:max_x,min_y:max_y]


def get_apn(cropped_lesion, cropped_lesion_mask ):
    """ Input: Croped version of the image and its mask that contains the major area of the lesion. 
        Output: The proportion of overlapp between the darker segmented area and the cropped original mask. """
    #flatten the image
    reshape=cropped_lesion.reshape(-1,3)
    reshape.shape

    #cluster all pixels into 2 segments
    k_clust=KMeans(n_clusters=2)
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

get_apn(cropped_lesion, cropped_lesion_mask)