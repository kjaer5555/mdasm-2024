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
import sys
import csv

class Tree:
    def __init__(self,n):
        self.n_trees = n
        self.depth_dict = dict()
        self.std_dict = dict()
        self.max_score=0
        self.max_depth=0
        
    def test_depth(self,d,X_scaled_test_data,y):
        print("number of trees: "+str(self.n_trees)+"\tdepth: "+str(d))
        rf_classifier = RandomForestClassifier(n_estimators=self.n_trees, max_depth=d, bootstrap=True)
        scores = cross_val_score(rf_classifier, X_scaled_test_data, y, cv=5, scoring='accuracy') #perform cv
        self.std_dict[d] = np.std(scores)
        self.depth_dict[scores.mean()]=d
    def get_best_score(self):
        self.max_score = max(self.depth_dict.keys())
        self.max_depth = self.depth_dict[self.max_score]
        return self.max_score

if len(sys.argv)>1:
    sti_features = sys.argv[1]
else:
    sti_features = 'train_data.csv'
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
rf_ds_list=list(range(1,4)) # creating list of depths for rf
best_rf_depth=0  # best depth found after cv

tree = Tree(4500)
for k in rf_ds_list:
    tree.test_depth(k,X_scaled_test_data,y)
with open("best_tree_extract.csv","w+") as file:
    for key,value in tree.depth_dict.items():
        file.write(str(key)+", "+str(value)+", "+str(tree.std_dict[value])+"\n")

