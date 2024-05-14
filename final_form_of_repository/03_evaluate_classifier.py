import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
import pandas as pd
from extract_features import extract_features
import sys
import os

def classify(img, mask):
    scaler = StandardScaler()
    #Extract features (the same ones that you used for training)
    X = extract_features(img, mask)
    X=X[['H_value', 'S_value', 'V_value', 'red_presence', 'brown_presence', 'blue_presence', 'pink_presence', 'white_presence','black_presence','atypical_pigment_network', 'blue-white_veil', 'asymmetry_values']]
    X = np.array(X) # here should be your X in np.array format
    #Use standard scaler to transform data
    X_scaled = scaler.fit_transform(X.T).T
    #print(X_scaled)
    #exit()
    
    model_log = pd.read_pickle('./models/groupR_log_regr_classifier.sav')
    model_random_forest = pd.read_pickle('./models/groupR_random_forest_classifier.sav')
    model_knn = pd.read_pickle('./models/groupR_knn_classifier.sav')
    pca_reload = pd.read_pickle('./models/pca.sav')
    
    X_scaled_pca=pca_reload.transform(X_scaled)
    #X_scaled_pca = pca_reload.transform(X_scaled)

    #prediction
    p1=model_log.predict_proba(X_scaled)
    p2=model_random_forest.predict_proba(X_scaled)
    p3=model_knn.predict_proba(X_scaled_pca)

    pred_prob=np.sum([p1[0][0],p2[0][0],p3[0][0]])/3
    pred_label=(pred_prob>0.5)
    
    return pred_label, pred_prob
 
if len(sys.argv)==3:
    img=sys.argv[1]
    mask=sys.argv[2]
    print(classify(img,mask))
else:
    print("No input")
    '''
    imgs=os.listdir("images")
    for i in imgs:
        imgpath="images/"+i
        maskpath="masks/"+i.split(".")[0]+"_mask."+i.split(".")[1]
        if os.path.exists(maskpath):
            print(classify(imgpath,maskpath))
    '''