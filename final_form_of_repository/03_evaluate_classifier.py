import pandas as pd
from extract_features import extract_features

#Combine logistic regression and random forest classifier
def classify(img, mask):
    scaler_reload, features = pd.read_pickle('./models/scaler.sav')
    model_log = pd.read_pickle('./models/groupR_log_regr_classifier.sav')
    model_random_forest = pd.read_pickle('./models/groupR_random_forest_classifier.sav')

    X = extract_features(img, mask)

    X = X[features]
    X_scaled = scaler_reload.transform(X)

    #prediction
    log=model_log.predict_proba(X_scaled)
    rf=model_random_forest.predict_proba(X_scaled)
    pred_prob=(rf[0][1]+log[0][1])/2

    threshold=0.5
    pred_label=(pred_prob>threshold)
    
    return pred_label, pred_prob

#Example: print(classify("img.png","mask.png"))