#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import scipy.io
import numpy as np

import matplotlib.pyplot as plt
from nilearn.connectome import ConnectivityMeasure
from scipy.signal import filtfilt


#README: This is a script with some (not so many) custom functions that I've been using during the master thesis. Some of them were finally not that useful, but you will find this functions imported in almost any of my scripts.

def import_mat(filename, var_name):
    '''Function that loads a mat file and gets one variable, obtaining a numpy array'''
    mat = scipy.io.loadmat(filename)
    return np.array(mat.get(var_name))


# Get the number of cells
def get_fMRI(dataset, variable):
    """
       Opens a matlab file, opens a variable of the file and then gets into each cell of this variable. Finally it 
       takes the fMRI-data matrix from each cell and appends it to a list, which transforms into an array
       
       Parameters:
        filename: string.mat
        variable: string
           variable of the file
        
        Return: 
         numpy 3D array
     """
    # Load the MATLAB file
    mat_file = scipy.io.loadmat(dataset)

    # Get the variable that contains the cells
    cell_var = mat_file[variable]
    
    n = cell_var.shape[1]

    matrix_list = []

    for i in range(n):
        if cell_var[0][i][0][0][0].shape[1] == 152:
            matrix_list.append(cell_var[0][i][0][0][0])
    
    # Convert the list of matrices into a numpy array
    matrix_array = np.stack(matrix_list, axis=0)

    return matrix_array


def get_FC(dataset):
    
    """Iterates over the subjects of a fMRI dataset and calculates the Pearson Correlation for each parcellation, obtaining the functional connectivity foreach 
    subject, which appends to a list later transformed into a numpy array.
    
    Parameters:
        dataset: numpy array
           dataset with fMRI data for x subjects
        
        
    Return: 
        numpy 3D array
    """
    FC = []
    for subject in range(dataset.shape[0]):
        data = np.transpose(dataset[subject])
        correlation_measure = ConnectivityMeasure(kind='correlation', discard_diagonal = True)
        correlation_matrix = correlation_measure.fit_transform([data])[0]
        FC.append(correlation_matrix)
    
    npFC = np.array(FC)
    return npFC


def plot_FC(dataset, subjects, nrows, ncols):
    
    """Iterates to plot the functional connectivity matrix (as a heatmap) for a desired number of subjects.
    
    Parameters:
        dataset: numpy ndimensional array
        subjects: int
           number of subjects we want to plot
        nrows: int
            number of rows for the subplot
        ncols: int¡
            number of cols for the subplot
    Return: 
        heatmaps
        """
    plt.figure(figsize=(15, 15))
    for subj in range(subjects):
        plt.subplot(nrows,ncols, subj+1)
        #Mask out the major diagonal
        np.fill_diagonal(dataset[subj], 0)
        plt.imshow(dataset[subj], interpolation="nearest", cmap="jet", vmax=1, vmin=-1)
        plt.colorbar(shrink = 0.2)
    
    plt.tight_layout()
    
    
def average_FC(matrices):
    # Initialize an empty matrix to store the voxel averages
    avg_matrix = np.zeros_like(matrices[0])
    
    # Iterate through each voxel in the matrices
    for i in range(matrices[0].shape[0]):
        for j in range(matrices[0].shape[1]):
                 # Sum the value of the voxel across all n matrices
                voxel_sum = 0
                for matrix in matrices:
                    voxel_sum += matrix[i][j]
                # Calculate the average and store it in the avg_matrix
                avg_matrix[i][j] = voxel_sum / matrices.shape[0]
    
    avg_fc = np.array(avg_matrix)
    return avg_fc

def butter_bandpass(lowcut, highcut, fs, order=5):
    return butter(order, [lowcut, highcut], fs=fs, btype='band')

def butter_lowpass_filter(data, low_cutoff, fs, order):
    normal_cutoff = low_cutoff/nyq 
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

def butter_highpass_filter(data, low_cutoff, fs, order):
    normal_cutoff = low_cutoff/nyq 
    b, a = butter(order, normal_cutoff, btype='highpass', analog=False)
    y = filtfilt(b, a, data)
    return y

def butter_bandpass_filter(data, low_cutoff, high_cutoff, fs, order):
    normal_cutoff = [low_cutoff/nyq, high_cutoff/nyq]
    # Get the filter coefficients 
    b, a = butter(order, normal_cutoff, btype='bandpass', analog=False)
    y = filtfilt(b, a, data)
    return y