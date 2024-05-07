
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold,GroupKFold
import os

class Models_validator:
    def __init__(self,x,y):
        self.x=x
        self.y=y
    def rf_parameters(self,tree_range,depth_range):
        tree_best_depth=dict()
        tree_best_accuracy=dict()
        for t in tree_range:
            score_depth=dict()
            for d in depth_range:
                rf_classifier = RandomForestClassifier(n_estimators=t, max_depth=d, bootstrap=True)
                scores = cross_val_score(rf_classifier, self.x, self.y, cv=5, scoring='accuracy') #perform cv
                score_depth[scores.mean()]=d
            best_accuracy=max(score_depth.keys())
            
            tree_best_accuracy[t]=best_accuracy
            tree_best_depth[t]=score_depth[best_accuracy]
        index=list(tree_best_accuracy.values()).index(max(tree_best_accuracy.values()))
        t=list(tree_best_accuracy.keys())[index]
        return t,tree_best_depth[t]
    
    def knn_parameters(self,neighbors_range):
        score_neighbors=dict()
        for n in neighbors_range:
            kn_classifier = KNeighborsClassifier(n_neighbors=n)
            scores = cross_val_score(kn_classifier, self.x, self.y, cv=5, scoring='accuracy') #perform cv
            score_neighbors[scores.mean()]=n
        
        return score_neighbors[max(score_neighbors.keys())]

#Where did we store the features?
file_features = 'train_75_people_data.csv'
feature_names = ['H_value', 'S_value', 'V_value', 'red_presence', 'brown_presence', 'blue_presence', 'pink_presence', 'white_presence','black_presence','atypical_pigment_network', 'blue-white_veil', 'asymmetry_values']

# Load the features - remember the example features are not informative
df_features = pd.read_csv(file_features)

# Make the dataset, you can select different classes (see task 0)
x = np.array(df_features[feature_names])
y =  df_features['cancer_or_not']   #now True means healthy nevus, False means something else
patient_id = df_features['person_id']


#Prepare cross-validation - images from the same patient must always stay together
num_folds = 5
group_kfold = GroupKFold(n_splits=num_folds)
group_kfold.get_n_splits(x, y, patient_id)
scaler = StandardScaler()
X_scaled_test_data = scaler.fit_transform(x)

rf_cv_scores = [] # creating list of cv scores
rf_ds_list=list(range(1,16)) # creating list of depths for rf
best_rf_depth=0  # best depth found after cv

model_validator = Models_validator(X_scaled_test_data,y)
rf_trees,rf_depth = model_validator.rf_parameters(range(500,1000,500),range(1,2))
knn_neighbors = model_validator.knn_parameters(range(1,30))

#Different classifiers to test out
classifiers = [
    KNeighborsClassifier(knn_neighbors),
    LogisticRegression(),
    RandomForestClassifier(n_estimators=rf_trees, max_depth=rf_depth, bootstrap=True)
]
num_classifiers = len(classifiers)

      
acc_val = np.empty([num_folds,num_classifiers])

for i, (train_index, val_index) in enumerate(group_kfold.split(x, y, patient_id)):
    
    x_train = x[train_index,:]
    y_train = y[train_index]
    x_val = x[val_index,:]
    y_val = y[val_index]
    
    
    for j, clf in enumerate(classifiers): 
        
        #Train the classifier
        clf.fit(x_train,y_train)
    
        #Evaluate your metric of choice (accuracy is probably not the best choice)
        acc_val[i,j] = accuracy_score(y_val, clf.predict(x_val))
   
    
#Average over all folds
average_acc = np.mean(acc_val,axis=0) 
   
print('Classifier 1 average accuracy={:.3f} '.format(average_acc[0]))
print('Classifier 2 average accuracy={:.3f} '.format(average_acc[1]))



#Let's say you now decided to use the 5-NN 
clf1  = KNeighborsClassifier(n_neighbors = 17)
clf2 = LogisticRegression()
clf3 =RandomForestClassifier(n_estimators=4500, max_depth=7, bootstrap=True)
#It will be tested on external data, so we can try to maximize the use of our available data by training on 
#ALL of x and y
aiakos = clf1.fit(x,y)
minos = clf2.fit(x,y)
rhadamanthys = clf3.fit(x,y)

#This is the classifier you need to save using pickle, add this to your zip file submission
model1 = 'groupR_knn_classifier.sav'
model2 = 'groupR_log_regr_classifier.sav'
model3 = 'groupR_random_forest_classifier.sav'
pickle.dump(aiakos, open(model1, 'wb'))
pickle.dump(minos, open(model2, 'wb'))
pickle.dump(rhadamanthys, open(model3, 'wb'))