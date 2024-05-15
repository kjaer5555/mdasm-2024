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
        self.scoring=('accuracy','recall','roc_auc')
    
    def get_scores(self,clf,p1,p2):
        if clf=='rf':
            classifier = RandomForestClassifier(n_estimators=p1, max_depth=p2, bootstrap=True,random_state=42)
            scores = cross_validate(classifier, self.x, self.y, cv=5, scoring=self.scoring)
        elif clf=='knn':
            pca = PCA(n_components=p1)
            pca_component = pca.fit_transform(self.x)
            classifier = KNeighborsClassifier(n_neighbors=p2)
            scores = cross_validate(classifier, pca_component , y, cv=5, scoring=self.scoring)
        return scores

    def maximize_score(self,clf,range1,range2,score_label="roc_auc",min_accuracy=0.0):
        p1_p2list=dict()
        
        p1_p2=dict()
        p1_score=dict()
        p1_accuracy=dict()
        print(f"Training: {clf}")
        for p1 in range1:
            print(f"progress: {round(100*p1/range1[-1],2)}%")
            p2_acc=dict()
            p2_score=dict()
            for p2 in range2:
                scores=self.get_scores(clf,p1,p2)
                accuracy=scores['test_accuracy'].mean()
                score=scores['test_'+score_label].mean()

                #roc_auc=scores['test_roc_auc'].mean()

                if accuracy>=min_accuracy:
                    p2_acc[p2]=accuracy
                    p2_score[p2]=score

            p1_p2list[p1]=p2_score.values()

            if len(p2_score.keys())>0:
                best_score=max(p2_score.values())
                p1_score[p1]=best_score
                p2_for_best_score=select_key(p2_score,best_score)
                p1_p2[p1]=p2_for_best_score
                p1_accuracy[p1]=p2_acc[p2_for_best_score]
        
        #df=pd.DataFrame.from_dict(p1_p2list, orient="index")
        #df.to_csv(f"{clf}_plot.csv")

        if len(p1_score.values())>0:
            maxscore=max(p1_score.values())
            p1=select_key(p1_score,maxscore)
            acc=p1_accuracy[p1]
            #print(maxscore)
            #print(acc)

            d = {"score":maxscore,"accuracy":acc,"p1":p1,"p2":p1_p2[p1]}
            return d
            #return p1,p1_p2[p1]
        return f"No model with accuracy at least {min_accuracy}"

#Where did we store the features?
file_features = 'features/train_75_people_data.csv'
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
X_scaled_train_data = scaler.fit_transform(df_features[feature_names])


rf_cv_scores = [] # creating list of cv scores
rf_ds_list=list(range(1,16)) # creating list of depths for rf
best_rf_depth=0  # best depth found after cv

model_validator = Models_validator(X_scaled_train_data,y)

#knn_results = model_validator.maximize_score('knn',range(3,4),range(1,27),score_label="roc_auc")
#rf_results = model_validator.maximize_score('rf',range(71,72),range(1,7),score_label="roc_auc")
knn_results = model_validator.maximize_score('knn',range(1,13),range(1,27),score_label="roc_auc")
rf_results = model_validator.maximize_score('rf',range(6,98,1),range(1,7),score_label="roc_auc")

trees=rf_results["p1"]
depth=rf_results["p2"]

rf_recall=rf_results["score"]
rf_accuracy=rf_results["accuracy"]

print(f"trees: {trees}, depth: {depth}, recall: {rf_recall}, accuracy: {rf_accuracy}")

dim=knn_results["p1"]
k=knn_results["p2"]
knn_recall=knn_results["score"]
knn_accuracy=knn_results["accuracy"]

print(f"dim: {dim}, k: {k}, recall: {knn_recall}, accuracy: {knn_accuracy}")

pca = PCA(n_components=dim)
pca_component = pca.fit_transform(X_scaled_train_data)
clf1  = KNeighborsClassifier(n_neighbors = k)
clf2 = LogisticRegression()
clf3 =RandomForestClassifier(n_estimators=trees, max_depth=depth, bootstrap=True)
#three judges
aiakos = clf1.fit(pca_component,y) #knn needs to fit for the pca component
minos = clf2.fit(X_scaled_train_data,y)
rhadamanthys = clf3.fit(X_scaled_train_data,y)

#This is the classifier you need to save using pickle, add this to your zip file submission
model1 = 'models/groupR_knn_classifier.sav'
model2 = 'models/groupR_log_regr_classifier.sav'
model3 = 'models/groupR_random_forest_classifier.sav'

pickle.dump(aiakos, open(model1, 'wb'))
pickle.dump(minos, open(model2, 'wb'))
pickle.dump(rhadamanthys, open(model3, 'wb'))
pickle.dump(pca,open('models/pca.sav','wb'))
pickle.dump((scaler, feature_names), open('models/scaler.sav','wb'))
