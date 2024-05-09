# -*- coding: utf-8 -*-
"""
Created on Mon May  6 14:17:25 2024

@author: beneitof
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
import seaborn as sns

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

dataset = SCZ.copy()
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

dataset = BP.copy()
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
BP_EC_scaled = all_data_normalized

dataset = ADHD.copy()
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
ADHD_EC_scaled = all_data_normalized


#Create dataset
    
SCZ = []
for i in range(0, SCZ_EC_scaled.shape[0]):
    SCZ.append(SCZ_EC_scaled[i])

BP = []
for i in range(0, BP_EC_scaled.shape[0]):
    BP.append(BP_EC_scaled[i])
    
ADHD = []
for i in range(0, ADHD_EC_scaled.shape[0]):
    ADHD.append(ADHD_EC_scaled[i])

    
EC_matrices = []
for i in range(0, len(SCZ)):
    EC_matrices.append(SCZ[i])
for i in range(0, len(BP)):
    EC_matrices.append(BP[i])
for i in range(0, len(ADHD)):
    EC_matrices.append(ADHD[i])

EC_matrices = np.array(EC_matrices)

EC_vectors = []
for i in range(0, EC_matrices.shape[0]):
    EC_vectors.append(EC_matrices[i].flatten())
EC_vectors = np.array(EC_vectors)

#We create a DF:
vectors_df = pd.DataFrame(EC_vectors)

SCZ_tag = np.array(['SCZ']*50)
BP_tag = np.array(['BP']*49)
ADHD_tag = np.array(['ADHD']*40)
tags = np.concatenate((SCZ_tag, BP_tag, ADHD_tag))

vectors_df['condition'] = tags
df = vectors_df


SCZ_df = df[df['condition']=='SCZ'].sample(50)
BP_df = df[df['condition']=='BP'].sample(49)
ADHD_df = df[df['condition']=='ADHD'].sample(40)



vectors_dfA = pd.concat([SCZ_df, BP_df, ADHD_df])


X = vectors_dfA.drop('condition', axis = 1).copy()
y = vectors_dfA['condition'].copy()

#-------------------------------------------------------------------
#-------------------------------------------------------------------
#CLASSIFIER

a = np.arange(0,50,1)
b = np.random.choice(a,50, replace=False)
intermediateSCZ_df = df[df['condition']=='SCZ']
SCZ_rand = intermediateSCZ_df.iloc[b]
SCZ_df = SCZ_rand[0:40]
    
a = np.arange(0,49,1)
b = np.random.choice(a,49, replace=False)
intermediateBP_df = df[df['condition']=='BP']
BP_rand = intermediateBP_df.iloc[b]
BP_df = BP_rand[0:40]

a = np.arange(0,40,1)
b = np.random.choice(a,40, replace=False)
intermediateADHD_df = df[df['condition']=='ADHD']
ADHD_rand = intermediateADHD_df.iloc[b]
ADHD_df = ADHD_rand[0:40]
    
vectors_dfA = pd.concat([SCZ_df, BP_df, ADHD_df])
    
X = vectors_dfA.drop('condition', axis = 1).copy()
y = vectors_dfA['condition'].copy()
X_train_scaled, X_test_scaled, y_train, y_test = train_test_split(X, y, test_size=0.20)

unique, counts = np.unique(y_train, return_counts=True)



if counts[0]/(counts[0]+counts[1]+counts[2]) < 0.32:
    print('again')
if counts[1]/(counts[0]+counts[1]+counts[2]) < 0.32:
    print('again')
if counts[2]/(counts[0]+counts[1]+counts[2]) < 0.32:
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
    cm = confusion_matrix(y_test, y_pred, labels=['SCZ', 'BP', 'ADHD'])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['SCZ', 'BP', 'ADHD'])
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

results = []
i = 0
while i < 200:
    a = np.arange(0,50,1)
    b = np.random.choice(a,50, replace=False)
    intermediateSCZ_df = df[df['condition']=='SCZ']
    SCZ_rand = intermediateSCZ_df.iloc[b]
    SCZ_df = SCZ_rand[0:40]
    
    a = np.arange(0,49,1)
    b = np.random.choice(a,49, replace=False)
    intermediateBP_df = df[df['condition']=='BP']
    BP_rand = intermediateBP_df.iloc[b]
    BP_df = BP_rand[0:40]

    a = np.arange(0,40,1)
    b = np.random.choice(a,40, replace=False)
    intermediateADHD_df = df[df['condition']=='ADHD']
    ADHD_rand = intermediateADHD_df.iloc[b]
    ADHD_df = ADHD_rand[0:40]
    
    vectors_dfA = pd.concat([SCZ_df, BP_df, ADHD_df])
    
    X = vectors_dfA.drop('condition', axis = 1).copy()
    y = vectors_dfA['condition'].copy()
    X_train_scaled, X_test_scaled, y_train, y_test = train_test_split(X, y, test_size=0.20)

    unique, counts = np.unique(y_train, return_counts=True)

    if counts[0]/(counts[0]+counts[1]+counts[2]) < 0.32:
        print('again')
    if counts[1]/(counts[0]+counts[1]+counts[2]) < 0.32:
        print('again')
    if counts[2]/(counts[0]+counts[1]+counts[2]) < 0.32:
        print('again')
    else:
        
        clf = RandomForestClassifier(criterion = grid_search.best_params_['criterion'],
                                     max_depth= grid_search.best_params_['max_depth'],
                                     min_samples_split= grid_search.best_params_['min_samples_split'],
                                     n_estimators = grid_search.best_params_['n_estimators']
                                     )

        clf.fit(X_train_scaled, y_train)
        y_pred = clf.predict(X_test_scaled)
        cm = confusion_matrix(y_test, y_pred, labels=['SCZ', 'BP', 'ADHD'])
        results.append(cm)
        print(f"iteration{i}")
        i += 1


plt.subplot(1,2,1)
plt.plot(RF_CTRL_classification_accuracy)
plt.subplot(1,2,2)
plt.plot(RF_SCZ_classification_accuracy)
plt.show()
print(np.mean(RF_CTRL_classification_accuracy))
print(np.mean(RF_SCZ_classification_accuracy))

np.save('FC-NS-SCZ BP ADHD', np.array(results), allow_pickle = True)

RF_mean = np.sum(results, axis = 0)

RF_final = np.zeros((3,3))
for i in range(0,3):
    for j in range(0,3):
        RF_final[i][j] = RF_mean[i][j]/np.sum(RF_mean[i])
        
sns.heatmap(RF_final, cmap = 'viridis', annot = True, vmax=np.max(RF_final),
           vmin=np.min(RF_final), linewidths = 0.5)

c = np.array(results)
c.shape