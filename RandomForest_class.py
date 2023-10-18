# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 15:52:32 2023

@author: javie
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import scale
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay



#Import data
BP_dist = np.load("C:/Users/javie/Desktop/results/resultsBP_dist.npy")
BP_EC = np.load("C:/Users/javie/Desktop/results/resultsBP_EC.npy")

CTRL_dist = np.load("C:/Users/javie/Desktop/results/resultsCTRL_dist.npy")
CTRL_EC = np.load("C:/Users/javie/Desktop/results/resultsCTRL_EC.npy") 
 

SCZ_dist = np.load("C:/Users/javie/Desktop/results/resultsSCZ_dist.npy")
SCZ_EC = np.load("C:/Users/javie/Desktop/results/resultsSCZ_EC.npy") 


ADHD_dist = np.load("C:/Users/javie/Desktop/results/resultsADHD_dist.npy")
ADHD_EC = np.load("C:/Users/javie/Desktop/results/resultsADHD_EC.npy") 


#I first normalize data 
#(but i think is unnecessary and the classification works without it, 
#was just part of different things i tried but i left it untouched):
    
dataset = BP_EC.copy()
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

dataset = CTRL_EC.copy()
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

dataset = SCZ_EC.copy()
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


dataset = ADHD_EC.copy()
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


#This is also an unnecessary step :) but made me easier to work. It is really unnecessary haha
BP = []
for i in range(0, 49):
    BP.append(BP_EC_scaled[i])
    
CTRL = []
for i in range(0,121):
    CTRL.append(CTRL_EC_scaled[i])
    
SCZ = []
for i in range(0,50):
    SCZ.append(SCZ_EC_scaled[i])
    
ADHD = []
for i in range(0, 40):
    ADHD.append(ADHD_EC_scaled[i])

#Now pool all the EC matrices together:
    
EC_matrices = []
for i in range(0, len(BP)):
    EC_matrices.append(BP[i])
for i in range(0, len(CTRL)):
    EC_matrices.append(CTRL[i])
for i in range(0, len(SCZ)):
    EC_matrices.append(SCZ[i])
for i in range(0, len(ADHD)):
    EC_matrices.append(ADHD[i])
EC_matrices = np.array(EC_matrices)

#And flatten them into vectors:
EC_vectors = []
for i in range(0, EC_matrices.shape[0]):
    EC_vectors.append(EC_matrices[i].flatten())
EC_vectors = np.array(EC_vectors)

#We create a DF (to add the tags later):
vectors_df = pd.DataFrame(EC_vectors)

#Add tags (manual step-to be optimized)
BP_tag = np.array(['BP']*49)
CTRL_tag = np.array(['CTRL']*121)
SCZ_tag = np.array(['SCZ']*50)
ADHD_tag = np.array(['ADHD']*40)
tags = np.concatenate((BP_tag, CTRL_tag, SCZ_tag, ADHD_tag))
vectors_df['condition'] = tags


#This was also part of a trial, I wanted to know if adding the distance between 
#simFC and empFC would add something to the classification (it didn´t and it didn´t make sense tbh)
#you can just delete these lines if you want

all_dist = np.concatenate((BP_dist,CTRL_dist, SCZ_dist, ADHD_dist))
vectors_df['distance'] = all_dist
df = vectors_df

#now sample according to the less representated group (I created sub df)
BP_df = df[df['condition']=='BP'].sample(40)
CTRL_df = df[df['condition']=='CTRL'].sample(40)
SCZ_df = df[df['condition']=='SCZ'].sample(40)
ADHD_df = df[df['condition']=='ADHD'].sample(40)

vectors_dfA = pd.concat([BP_df, CTRL_df, SCZ_df, ADHD_df])

vectors_dfA.drop('distance', axis = 1)
vectors_dfA.columns = vectors_df.columns.astype(str)

#Get the EC vectors and separatedly the tags

X = vectors_dfA.drop('condition', axis = 1).copy()
y = vectors_dfA['condition'].copy()


#-------------------------------------------------------------------
#-------------------------------------------------------------------
#CLASSIFIER

#First I do the hyperparameter tunning

#obtain the training and test datasets
X_train_scaled, X_test_scaled, y_train, y_test = train_test_split(X, y, test_size=0.20)
unique, counts = np.unique(y_train, return_counts=True) 
unique2, counts2 = np.unique(y_test, return_counts=True) 

#Now I re-define the test dataset by includding all the individuals that were not included first
#If you notice, we sampled only for 40individuals (max number ADHD), but the test will have all the remaining subjects as well
R = X_train_scaled.index
test_dataset = vectors_df.drop(R)
test_dataset.columns = test_dataset.columns.astype(str)
X_test_scaled = test_dataset.drop('condition', axis = 1).copy()
y_test = test_dataset['condition'].copy()

#Now I 'force' the training sampling to be more or less equal, otherwise the classification is heavily affected/biased
if counts[0]/(counts[0]+counts[1]+counts[2]+counts[3]) < 0.24:
    print('again')
if counts[1]/(counts[0]+counts[1]+counts[2]+counts[3]) < 0.24:
    print('again')
if counts[2]/(counts[0]+counts[1]+counts[2]+counts[3]) < 0.24:
    print('again')
if counts[3]/(counts[0]+counts[1]+counts[2]+counts[3]) < 0.24:
    print('again')
else:
    #do the Random Forest (subjective choice of parameters)
    clf = RandomForestClassifier(criterion = 'gini',
                                 max_depth= 10,
                                 min_samples_split=10,
                                 n_estimators = 200
                                 )
    
    clf.fit(X_train_scaled, y_train)
    
    clf.feature_importances_   
    
    y_pred = clf.predict(X_test_scaled)
    print(y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=['CTRL', 'SCZ', 'BP', 'ADHD'])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['CTRL', 'SCZ', 'BP', 'ADHD'])
    disp.plot()
    plt.show()
    
    print(f'accuracy is {accuracy_score(y_test, y_pred)}')
    
    print('CROSSVALIDATION:')
    print(cross_val_score(clf, X_train_scaled, y_train, cv = 20))
    
    print(classification_report(y_test, y_pred))

#hyperparameter tunning
n_est = range(10, 160, 25)
depth = range(1, 21, 2)
splits = range(1, 16, 3)
param_grid = {
    'n_estimators': n_est,
    'max_depth': depth,
    'min_samples_split': splits,
}

grid_search = GridSearchCV(estimator = clf, param_grid=param_grid, cv=10)
grid_search.fit(X_train_scaled, y_train)


print("Best Hyperparameters: ", grid_search.best_params_)
print("Best Score: ", grid_search.best_score_)

best_model = grid_search.best_estimator_
accuracy = best_model.score(X_test_scaled, y_test)
print("Test Set Accuracy: ", accuracy)

#Once got the hyperparameters you can go to obtain the results. It is basically a repetition of the 
#lines above. Be careful, there are some manual steps, I will indicate (optimise if you want :)


results = []
i = 0
#define number of iterations
while i < 100:
    CTRL_df = df[df['condition']=='CTRL'].sample(40)
    SCZ_df = df[df['condition']=='SCZ'].sample(40)
    BP_df = df[df['condition']=='BP'].sample(40)
    ADHD_df = df[df['condition']=='ADHD']
    vectors_dfA = pd.concat([CTRL_df, SCZ_df, BP_df, ADHD_df],ignore_index=True)
    vectors_dfA.drop('distance', axis = 1)
    vectors_dfA.columns = vectors_dfA.columns.astype(str)
    X = vectors_dfA.drop('condition', axis = 1).copy()
    y = vectors_dfA['condition'].copy()
    X_train_scaled, X_test_scaled, y_train, y_test = train_test_split(X, y, test_size=0.20, shuffle = True)
    
    R = X_train_scaled.index
    test_dataset = vectors_df.drop(R)
    test_dataset.columns = test_dataset.columns.astype(str)
    X_test_scaled = test_dataset.drop('condition', axis = 1).copy()
    y_test = test_dataset['condition'].copy()
    unique, counts = np.unique(y_train, return_counts=True)
    if counts[0]/(counts[0]+counts[1]+counts[2]+counts[3]) < 0.24:
        print('again')
    if counts[1]/(counts[0]+counts[1]+counts[2]+counts[3]) < 0.24:
        print('again')
    if counts[2]/(counts[0]+counts[1]+counts[2]+counts[3]) < 0.24:
        print('again')
    if counts[3]/(counts[0]+counts[1]+counts[2]+counts[3]) < 0.24:
        print('again')
    else:
        #Here I am doing the hyperparameter for each case, you can just use the parameters that were obtained before
        #Again this was a sort of test for improving classification, but nothing will change drastically, only the 
        #speed of your script :)
        clf_svm = SVC()
        clf_svm.fit(X_train_scaled, y_train)
        
        param_grid = [
            {'C':[0.5, 1, 10, 100, 1000, 10000],
             'gamma':['scale', 1, 0.1, 0.01, 0.001, 0.0001, 0.00001],
             'kernel':['rbf']},
            ]
    
        optimal_params = GridSearchCV(
            SVC(),
            param_grid,
            cv = 10, 
            scoring = ['accuracy', 'f1_micro', 'f1_macro'],
            refit = 'accuracy',
            verbose = 0
            )
    
        optimal_params.fit(X_train_scaled, y_train)
    
    
        clf_svm = SVC(C = optimal_params.best_params_['C'], gamma = optimal_params.best_params_['gamma'])
        clf_svm.fit(X_train_scaled, y_train)
        
        #Now I save in the results the confusion matrix for each step, in the analysis + final plotting you should average them
        y_pred = clf_svm.predict(X_test_scaled)
        cm = confusion_matrix(y_test, y_pred, labels=['CTRL', 'SCZ', 'BP', 'ADHD'])
        results.append(cm)
        print(f"iteration{i}")
        i += 1


np.save('SVM_cm', np.array(results), allow_pickle = True)















