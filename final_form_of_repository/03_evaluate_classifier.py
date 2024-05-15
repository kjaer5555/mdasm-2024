import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
import pandas as pd
from extract_features import extract_features
import sys
import os

def classify(img, mask):
    scaler_reload, features = pd.read_pickle('./models/scaler.sav')
    model_log = pd.read_pickle('./models/groupR_log_regr_classifier.sav')
    model_random_forest = pd.read_pickle('./models/groupR_random_forest_classifier.sav')
    model_knn = pd.read_pickle('./models/groupR_knn_classifier.sav')
    pca_reload = pd.read_pickle('./models/pca.sav')

    #Extract features (the same ones that you used for training)
    X = extract_features(img, mask)

    X = X[features]
    X_scaled = scaler_reload.transform(X)
    X_scaled_pca=pca_reload.transform(X_scaled)


    #prediction
    p1=model_log.predict_proba(X_scaled)
    p2=model_random_forest.predict_proba(X_scaled)
    p3=model_knn.predict_proba(X_scaled_pca)

    pred_prob=np.sum([p1[0][1],p2[0][1],p3[0][1]])/3
    pred_label=(pred_prob>0.5)
    
    return pred_label, pred_prob
 
if len(sys.argv)==3:
    img=sys.argv[1]
    mask=sys.argv[2]
    print(classify(img,mask))
else:
    print("No input")
    test_csv=pd.read_csv("../test_25_people_data.csv")
    filenames=test_csv["Name_Of_Picture"].to_list()
    y=test_csv["cancer_or_not"].to_list()

    imgs=os.listdir("../images")
    cancer_counter=0
    all_counter=0
    for i in range(len(filenames)):
        
        filename=filenames[i]
        true_value=y[i]

        print(i/len(filenames))

        imgpath="../images/"+filename+".png"
        maskpath="../masks/"+filename+"_mask.png"
        if os.path.exists(maskpath):
            cancer,prob=classify(imgpath,maskpath)
            if cancer==true_value:
                cancer_counter+=1
            all_counter+=1
        else:
            print("nomask")
    print("Accuracy: ",cancer_counter/all_counter)
    