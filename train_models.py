
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
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
import os

class Models_validator:
    def __init__(self,x,y):
        self.x=x
        self.y=y
        self.acc_dict_tree = dict()
        self.rec_dict_tree = dict()
        self.pre_dict_tree = dict()
        self.f1_dict_tree = dict()
        self.std_dict_tree = dict()
        self.max_score=0
        self.max_depth=0
        self.acc_dict_knn = dict()
        self.rec_dict_knn = dict()
        self.pre_dict_knn = dict()
        self.f1_dict_knn = dict()
        self.acc_dict_logic = dict()
        self.rec_dict_logic = dict()
        self.pre_dict_logic = dict()
        self.f1_dict_logic = dict()
        
    def test_depth(self,trees):
        rf_ds_range=range(1,16)
        for d in rf_ds_range:
            rf_classifier = RandomForestClassifier(n_estimators=trees, max_depth=d, bootstrap=True)
            scoring = {'acc': 'accuracy',
            'rec': 'recall',
            'prec': 'precision',
            'f1_score':'f1'}
            scores = cross_validate(rf_classifier,X_scaled_test_data, y,cv=5, scoring=scoring)
            accuracy=scores['test_acc']
            recall=scores['test_rec']
            precision=scores['test_prec']
            f1_sore = scores['test_f1_score']
            self.std_dict[d] = np.std(accuracy)
            self.acc_dict[d]=accuracy.mean()
            self.pre_dict[d]=precision.mean()
            self.rec_dict[d]=recall.mean()
            self.f1_dict[d]=f1_sore.mean()
        
    def get_best_acc(self):
        self.max_score = max(self.acc_dict.values())
        self.max_depth = list(self.acc_dict.keys())[list(self.acc_dict.values()).index(self.max_score)]
        return self.max_score
    def get_best_acc(self):
        self.max_score = max(self.acc_dict.values())
        self.max_depth = list(self.acc_dict.keys())[list(self.acc_dict.values()).index(self.max_score)]
        return self.max_score
    def get_best_acc(self):
        self.max_score = max(self.acc_dict.values())
        self.max_depth = list(self.acc_dict.keys())[list(self.acc_dict.values()).index(self.max_score)]
        return self.max_score



rf_cv_scores = [] # creating list of cv scores
rf_ds_list=list(range(1,16)) # creating list of depths for rf
best_rf_depth=0  # best depth found after cv

tree = Tree(4500)
for k in rf_ds_list:
    tree.test_depth(k,X_scaled_test_data,y)
with open("best_tree_extract.csv","w+") as file:
    file.write("depth,accuracy,std_accuracy,recall,precision,f1_score\n")
    for d in tree.acc_dict.keys():
        acc=str(tree.acc_dict[d])
        prec=str(tree.pre_dict[d])
        rec=str(tree.rec_dict[d])
        f1=str(tree.f1_dict[d])
        std=str(tree.std_dict[d])
        file.write("{},{},{},{},{},{}\n".format(d,acc,std,rec,prec,f1))



#Where did we store the features?
file_features = 'features/features.csv'
feature_names = ['H_value', 'S_value', 'V_value', 'red_presence', 'brown_presence', 'blue_presence', 'pink_presence', 'white_presence','black_presence','atypical_pigment_network', 'blue-white_veil', 'asymmetry_values']

# Load the features - remember the example features are not informative
df_features = pd.read_csv(file_features)
label=df_features['cancer_or_not']

# Make the dataset, you can select different classes (see task 0)
x = np.array(df_features[feature_names])
y =  label == '1'   #now True means healthy nevus, False means something else
patient_id = df['patient_id']


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
tree.test_depth(4500)
#Different classifiers to test out
classifiers = [
    KNeighborsClassifier(27),
    LogisticRegression(),
    RandomForestClassifier(n_estimators=4500, max_depth=7, bootstrap=True)
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