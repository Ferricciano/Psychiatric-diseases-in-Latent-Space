# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

CTRL = np.load('C:/Users/beneitof/Desktop/FC comparison/CTRL.npy')
SCZ = np.load('C:/Users/beneitof/Desktop/FC comparison/SCZ.npy')
BP = np.load('C:/Users/beneitof/Desktop/FC comparison/BP.npy')
ADHD = np.load('C:/Users/beneitof/Desktop/FC comparison/ADHD.npy')

dataset = CTRL.copy()
sz_all = dataset.shape
all_data_normalized = dataset
for patient_num in range(0, sz_all[0]):
    patient = dataset[patient_num][:][:]
    norm_patient = patient
    mean_patient = np.mean(norm_patient)
    std_patient = np.std(norm_patient)
    for roi_num in range(0, sz_all[1]):
        norm_patient[roi_num][:] = (norm_patient[roi_num][:] - mean_patient)/std_patient
    all_data_normalized[patient_num][:][:] = norm_patient
CTRL_EC_scaled = all_data_normalized

dataset = np.concatenate((SCZ, BP, ADHD), axis = 0)
sz_all = dataset.shape
all_data_normalized = dataset
for patient_num in range(0, sz_all[0]):
    patient = dataset[patient_num][:][:]
    norm_patient = patient
    mean_patient = np.mean(norm_patient)
    std_patient = np.std(norm_patient)
    for roi_num in range(0, sz_all[1]):
        norm_patient[roi_num][:] = (norm_patient[roi_num][:] - mean_patient)/std_patient
    all_data_normalized[patient_num][:][:] = norm_patient
SCZ_EC_scaled = all_data_normalized


#Create dataset
CTRL = []
for i in range(0, 121):
    CTRL.append(CTRL_EC_scaled[i])
    
SCZ = []
for i in range(0, 136):
    SCZ.append(SCZ_EC_scaled[i])
    

    
EC_matrices = []
for i in range(0, len(CTRL)):
    EC_matrices.append(CTRL[i])
for i in range(0, len(SCZ)):
    EC_matrices.append(SCZ[i])

EC_matrices = np.array(EC_matrices)

EC_vectors = []
for i in range(0, EC_matrices.shape[0]):
    EC_vectors.append(EC_matrices[i].flatten())
EC_vectors = np.array(EC_vectors)

#We create a DF:
vectors_df = pd.DataFrame(EC_vectors)

CTRL_tag = np.array(['CTRL']*121)
SCZ_tag = np.array(['SCZ']*136)
tags = np.concatenate((CTRL_tag, SCZ_tag))

vectors_df['condition'] = tags
df = vectors_df


CTRL_df = df[df['condition']=='CTRL'].sample(121)
SCZ_df = df[df['condition']=='SCZ'].sample(121)



vectors_dfA = pd.concat([CTRL_df, SCZ_df])


X = vectors_dfA.drop('condition', axis = 1).copy()
y = vectors_dfA['condition'].copy()

#-------------------------------------------------------------------
#-------------------------------------------------------------------
#CLASSIFIER

a = np.arange(0,121,1)
b = np.random.choice(a,121, replace=False)
intermediateCTRL_df = df[df['condition']=='CTRL']
CTRL_rand = intermediateCTRL_df.iloc[b]
CTRL_df = CTRL_rand[0:100]

a = np.arange(0,136,1)
b = np.random.choice(a,136, replace=False)
intermediateSCZ_df = df[df['condition']=='SCZ']
SCZ_rand = intermediateSCZ_df.iloc[b]
SCZ_df = SCZ_rand[0:100]


vectors_dfA = pd.concat([CTRL_df, SCZ_df])
X = vectors_dfA.drop('condition', axis = 1).copy()
y = vectors_dfA['condition'].copy()
X_train_scaled, X_test_scaled, y_train, y_test = train_test_split(X, y, test_size=0.20, shuffle = True)

unique, counts = np.unique(y_train, return_counts=True)



if counts[0]/(counts[0]+counts[1]) < 0.48:
    print('again')
if counts[1]/(counts[0]+counts[1]) < 0.48:
    print('again')
else:
    
    clf = RandomForestClassifier(criterion = 'gini',
                                 max_depth= 11,
                                 min_samples_split=5,
                                 n_estimators = 60
                                 )
    
    clf.fit(X_train_scaled, y_train)
    
    clf.feature_importances_   
    
    y_pred = clf.predict(X_test_scaled)
    print(y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=['CTRL', 'SCZ'])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['CTRL', 'DISEASE'])
    disp.plot()
    plt.show()
    
    print(f'accuracy is {accuracy_score(y_test, y_pred)}')
    
    print('CROSSVALIDATION:')
    print(cross_val_score(clf, X_train_scaled, y_train, cv = 5))
    
    print(classification_report(y_test, y_pred))


    
 
n_est = range(20, 120, 25)
depth = range(1, 9, 2)
splits = (3,5,7,9)
criterion = ('gini', 'entropy', 'log_loss')
param_grid = {
    'n_estimators': n_est,
    'max_depth': depth,
    'min_samples_split': splits,
    'criterion': criterion,
}


grid_search = GridSearchCV(estimator = clf, param_grid=param_grid, cv=5)
grid_search.fit(X_train_scaled, y_train)


print("Best Hyperparameters: ", grid_search.best_params_)
print("Best Score: ", grid_search.best_score_)

best_model = grid_search.best_estimator_
accuracy = best_model.score(X_test_scaled, y_test)
print("Test Set Accuracy: ", accuracy)


RF_CTRL_classification_accuracy = []
RF_SCZ_classification_accuracy = []
overall_performance = []


i = 0
while i < 200:
    a = np.arange(0,121,1)
    b = np.random.choice(a,121, replace=False)
    intermediateCTRL_df = df[df['condition']=='CTRL']
    CTRL_rand = intermediateCTRL_df.iloc[b]
    CTRL_df = CTRL_rand[0:121]
    
    a = np.arange(0,136,1)
    b = np.random.choice(a,136, replace=False)
    intermediateSCZ_df = df[df['condition']=='SCZ']
    SCZ_rand = intermediateSCZ_df.iloc[b]
    SCZ_df = SCZ_rand[0:121]
  
    
    vectors_dfA = pd.concat([CTRL_df, SCZ_df])
    X = vectors_dfA.drop('condition', axis = 1).copy()
    y = vectors_dfA['condition'].copy()
    X_train_scaled, X_test_scaled, y_train, y_test = train_test_split(X, y, test_size=0.20, shuffle = True)
    
    

    unique, counts = np.unique(y_train, return_counts=True)
    if counts[0]/(counts[0]+counts[1]) < 0.48:
        print('again')
    if counts[1]/(counts[0]+counts[1]) < 0.48:
        print('again')
    else:
        
        clf = RandomForestClassifier(criterion = grid_search.best_params_['criterion'],
                                     max_depth= grid_search.best_params_['max_depth'],
                                     min_samples_split= grid_search.best_params_['min_samples_split'],
                                     n_estimators = grid_search.best_params_['n_estimators']
                                     )

        clf.fit(X_train_scaled, y_train)
        y_pred = clf.predict(X_test_scaled)
        cm = confusion_matrix(y_test, y_pred, labels=['CTRL', 'SCZ'])
        RF_CTRL_classification_accuracy.append(cm[0,0]/(cm[0,0]+cm[0,1]))
        RF_SCZ_classification_accuracy.append(cm[1,1]/(cm[1,0]+cm[1,1]))
        print(f"iteration{i}")
        i += 1


plt.subplot(1,2,1)
plt.plot(RF_CTRL_classification_accuracy)
plt.subplot(1,2,2)
plt.plot(RF_SCZ_classification_accuracy)
plt.show()
print(np.mean(RF_CTRL_classification_accuracy))
print(np.mean(RF_SCZ_classification_accuracy))

np.save('accuracy CTRL', np.array(RF_CTRL_classification_accuracy), allow_pickle = True)
np.save('accuracy DIS', np.array(RF_SCZ_classification_accuracy), allow_pickle = True)