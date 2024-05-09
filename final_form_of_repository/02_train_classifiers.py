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
from sklearn.model_selection import cross_val_score,cross_validate
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold,GroupKFold
from sklearn.decomposition import PCA

def select_key(d,value):
    index=list(d.values()).index(value)
    key=list(d.keys())[index]
    return key

class Models_validator:
    def __init__(self,x,y):
        self.x=x
        self.y=y
        self.scoring=('accuracy','recall')
    
    def get_scores(self,clf,p1,p2):
        if clf=='rf':
            classifier = RandomForestClassifier(n_estimators=p1, max_depth=p2, bootstrap=True)
            scores = cross_validate(classifier, self.x, self.y, cv=5, scoring=self.scoring)
        elif clf=='knn':
            pca = PCA(n_components=p1)
            pca_component = pca.fit_transform(self.x)
            classifier = KNeighborsClassifier(n_neighbors=p2)
            scores = cross_validate(classifier, pca_component , y, cv=5, scoring=self.scoring)
        return scores

    def train_model(self,clf,range1,range2,min_accuracy=0.7):
        p1_p2=dict()
        p1_recall=dict()
        p1_accuracy=dict()
        for p1 in range1:
            print(p1)
            p2_acc=dict()
            p2_rec=dict()
            for p2 in range2:
                scores=self.get_scores(clf,p1,p2)
                accuracy=scores['test_accuracy'].mean()
                recall=scores['test_recall'].mean()

                if accuracy>=min_accuracy:
                    p2_acc[p2]=accuracy
                    p2_rec[p2]=recall
            
            #best_accuracy=max(acc_depth.keys())
            if len(p2_rec.keys())>0:
                best_recall=max(p2_rec.values())
                p1_recall[p1]=best_recall
                p2_for_best_recall=select_key(p2_rec,best_recall)
                p1_p2[p1]=p2_for_best_recall
                p1_accuracy[p1]=p2_acc[p2_for_best_recall]
        
        if len(p1_recall.values())>0:
            maxrec=max(p1_recall.values())
            p1=select_key(p1_recall,maxrec)
            acc=p1_accuracy[p1]
            print(maxrec)
            print(acc)
            return p1,p1_p2[p1]
        return f"No model with accuracy at least {min_accuracy}"

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
#rf_trees,rf_depth = model_validator.train_model('rf',range(6,127,5),range(10,11))
#print(rf_trees,rf_depth)
#print()
pca_dimensions,knn_neighbors = model_validator.train_model('knn',range(1,13),range(1,27),min_accuracy=0.67)

print(pca_dimensions,knn_neighbors)
print()
exit()
#Let's say you now decided to use the 5-NN 
pca = PCA(n_components=pca_dimensions)
clf1  = KNeighborsClassifier(n_neighbors = knn_neighbors)
clf2 = LogisticRegression()
clf3 =RandomForestClassifier(n_estimators=rf_trees, max_depth=rf_depth, bootstrap=True)
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
pickle.dump(pca,open('pca.sav','wb'))
