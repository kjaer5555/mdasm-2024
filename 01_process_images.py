import os
import pandas as pd
# Import our own file that has the feature extraction functions
from extract_features import extract_features

#Where is the raw data
directory_images = "pictures + masks/images/"
directory_mask = "pictures + masks/masks/"
directory_metadata = '../metadata.csv'
images = os.listdir(directory_images)
image_extensions=dict()

df_features = pd.DataFrame(columns=['Name_Of_Picture','asymmetry_values','H_value','S_value','V_value','red_presence','brown_presence','blue_presence','pink_presence','black_presence','white_presence','atypical_pigment_network','blue-white_veil'])
n=len(images)

for i in range(n):
    image=images[i]
    imagename=image.split(".")[0]
    image_extension=image.split(".")[1]
    image_extensions[imagename]=image_extension
    imagepath=directory_images+image
    maskpath=directory_mask+imagename+'_mask.'+image_extension
    if os.path.exists(maskpath):
        row=extract_features(imagepath,maskpath)
        df_features=pd.concat([df_features,row])
    print("{}/{}".format(i+1,n))

#get original diagnostics
path_diagnostics_data = directory_metadata

df_img_diadnostics = pd.read_csv(path_diagnostics_data)

diagnostic = []
cancer = []

# Create a dictionary from df_img for quick lookup
img_dict = {img_id: diag for img_id, diag in zip(df_img_diadnostics['img_id'], df_img_diadnostics['diagnostic'])}

for name in df_features['Name_Of_Picture']:
    img_name = name +"."+ image_extensions[name]
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
df_features.to_csv('features/test_feature_extraction.csv')