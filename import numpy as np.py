import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
#from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
#from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score

sti_features = 'mdasm-2024/train_data.csv'
#sti_features ='features_optimized_1.csv'
#sti_features ='features_optimized_2.csv'
#sti_features ='features_optimized_3.csv'

data = pd.read_csv(sti_features)
feature_columns= ['H_value', 'S_value', 'V_value', 'red_presence', 'brown_presence', 'blue_presence', 'pink_presence', 'white_presence','black_presence','atypical_pigment_network', 'blue-white_veil', 'asymmetry_values']
#feature_columns = ['H_value', 'V_value', 'white_presence', 'blue-white_veil', 'asymmetry_values']
X = data[feature_columns]
y = data['cancer_or_not']

scaler = StandardScaler()
X_scaled_test_data = scaler.fit_transform(X)

rf_cv_scores = [] # creating list of cv scores
rf_ds_list=list(range(1,25)) # creating list of depths for rf
best_rf_depth=0  # best depth found after cv

rf_nt=list(range(500, 5000, 500))
best_score=0
tree_depth=dict()
depth_score=dict()
for tree_n in rf_nt:
    depth_dict = dict() 
    for d in rf_ds_list:
       
        rf_classifier = RandomForestClassifier(n_estimators=tree_n, max_depth=d, bootstrap=True)
        scores = cross_val_score(rf_classifier, X_scaled_test_data, y, cv=5, scoring='accuracy') #perform cv
        rf_cv_scores.append(scores.mean())
        if rf_cv_scores[-1]>best_score: #find best score parameters
            best_score=rf_cv_scores[-1]
            best_rf_depth=d
        depth_dict[best_score]=best_rf_depth
    max_score=max(depth_dict.keys())
    max_depth=depth_dict[max_score]
    tree_depth[max_depth]=tree_n
    depth_score[max_score]=max_depth

max_score=max(depth_score.keys())
max_depth=depth_score[max_score]
max_tree=tree_depth[max_depth]

values=[max_tree,max_depth,max_score]
print(values)

#max_tree = [tree_dict.index(row)+row for row in tree_dict.values() if row[1]==max_score][0]
#print(max_tree)
#train model with best found parameters and save it
#final_tree=RandomForestClassifier(n_estimators=1000, max_depth=best_rf_depth, bootstrap=True)
#final_tree.fit(X_scaled_test_data, y)

#with open(f'depth{best_rf_depth}_tree.pkl', 'wb') as f:
 #   pickle.dump(final_tree, f)

#print(f"depth{best_rf_depth}_tree saved successfully!")

#save picture, save csv, model
