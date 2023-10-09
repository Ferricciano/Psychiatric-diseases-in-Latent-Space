import matplotlib as mpl
import scipy.io as sio
import matplotlib.pyplot as plt
from autoEncoderDense import AutoEncoderDense
import numpy as np
from sklearn.model_selection import KFold
from tensorflow.keras import backend
import myFunctions as fct

# Define constant parameter values for training
LEARNING_RATE = 0.0001
BATCH_SIZE = 5
EPOCHS = 5

# Load your data, dimensions are:
# *Datasets sizes* sz[0] = num_subjects / sz[1] = num_ROIs / sz[2] = num_timepoints
# Dataset is called all_data here

all_data = fct.get_fMRI('ucla_controls_dbs80.mat', 'subject')

# Timestep
dt = 2  # in seconds

sz_all = all_data.shape
# Normalize and center at 0 all dataset
all_data_normalized = all_data
for patient_num in range(0, sz_all[0]):
    patient = all_data[patient_num, :, :]
    norm_patient = patient
    mean_patient = np.mean(norm_patient)
    std_patient = np.std(norm_patient)
    for roi_num in range(0, sz_all[1]):
        norm_patient[roi_num, :] = (norm_patient[roi_num, :] - mean_patient)/std_patient
    all_data_normalized[patient_num, :, :] = norm_patient

# This is how I built the data sets for test and training, "dataset" is your data, "chunk_siz" is the time window we
# talked about (~0.8-1sec for you --> you have to put it in number of frames),
# "train_prop" is the training vs test ratio

def build_training_test_sets(dataset, chunk_siz=60, train_prop=0.9):
    dataset = dataset.reshape(dataset.shape + (1,))
    sz_dat = dataset.shape
    print(sz_dat)
    tot_patients = sz_dat[0]
    num_chunks = int(np.floor(sz_dat[2]/chunk_siz))
    num_train_patients = int(np.floor(tot_patients*train_prop))
    x_train = np.empty([num_train_patients*num_chunks, sz_dat[1], chunk_siz, 1])
    x_test = np.empty([(tot_patients-num_train_patients)*num_chunks, sz_dat[1], chunk_siz, 1])
    for chunk_idx in range(0, num_chunks-1):
        # Take num_train_patients patients randomly in time chunk chunk_idx and add them to the training
        patients_to_add = np.random.choice(tot_patients, size=num_train_patients, replace=False)
        other_patients = np.delete(range(0, tot_patients), patients_to_add, axis=0)
        x_train[chunk_idx*num_train_patients:(chunk_idx+1)*num_train_patients, :, :] = \
            dataset[patients_to_add, :, chunk_idx*chunk_siz:(chunk_idx+1)*chunk_siz, :]
        x_test[chunk_idx*(tot_patients-num_train_patients):(chunk_idx+1)*(tot_patients-num_train_patients), :, :] = \
            dataset[other_patients, :, chunk_idx*chunk_siz:(chunk_idx+1)*chunk_siz, :]
    return x_train, x_test

# This is to train the AE
def train(x_train, x_valid, learning_rate, batch_size, epochs, latent_dim=10, chunk_siz=60):
    autoenc = AutoEncoderDense(
        input_size=[214, chunk_siz, 1],
        layers_dim=[128, 64, 32],
        latent_space_dim=latent_dim
    )
    autoenc.summary()
    autoenc.train(x_train, x_valid, learning_rate, batch_size, epochs)
    return autoenc


if __name__ == "__main__":
    # Define data set
    dataset, x_test = build_training_test_sets(all_data_normalized, 1, 1)
    sz_dat = dataset.shape
    print(f"dataset shape {sz_dat}")

    mean_val_mse = []
    mean_val_acc = []

    latdim_range = range(2, 20, 1)
    for lat_dim in latdim_range:
        VALIDATION_ACC = []
        VALIDATION_MSE = []

        # K-fold validation datasets
        k = 10
        kf = KFold(n_splits=k, shuffle=True, random_state=None)
        fold_iteration = 1
        for train_idx, test_idx in kf.split(dataset):
            print(f"Train idx = {train_idx} test idx = {test_idx}")
            x_train = dataset[train_idx, :, :, :]
            x_valid = dataset[test_idx, :, :, :]
            print(f"training set size={x_train.shape}")
            print(f"test set size={x_valid.shape}")

            # Training step # fold_iteration/k
            save_dir = "KFOLD_validation"

            autoencoder = train(x_train, x_valid, LEARNING_RATE, BATCH_SIZE, EPOCHS, lat_dim, 1)
            autoencoder.save(save_dir, f"model_{fold_iteration}_{lat_dim}")
            autoencoder.save_encoder(save_dir, f"encoder_{fold_iteration}_{lat_dim}")

            autoencoder_trained = AutoEncoderDense.load(save_dir, f"model_{fold_iteration}_{lat_dim}")
            autoencoder_trained.summary()
            weights = autoencoder_trained.get_weights()

            encoder_trained = AutoEncoderDense.load_encoder(save_dir, f"encoder_{fold_iteration}_{lat_dim}")
            encoder_trained.summary()
            encoder_weights = encoder_trained.get_weights()

            # Evaluate the performance of the model - VALIDATION SET
            x_predict = autoencoder_trained.predict(x_valid)
            results = autoencoder_trained.evaluate(x_predict, x_valid)

            print(f"MSE results: {results[1]}")
            print(f"accuracy results: {results[2]}")
            print(f"metrics names {autoencoder_trained.metrics_names}")

            VALIDATION_MSE.append(results[1])
            VALIDATION_ACC.append(results[2])

            backend.clear_session()

            fold_iteration += 1
            print(f"fold iteration # {fold_iteration}  lat dim {lat_dim}")
            print(f"MSE (valid) {VALIDATION_MSE}, lat dim {lat_dim}")

        mean_val_mse.append(np.mean(VALIDATION_MSE))
        mean_val_acc.append(np.mean(VALIDATION_ACC))

    # Save results evaluation for matlab
    sio.savemat(save_dir + f"/validation_mse_{lat_dim}.mat", mdict={'validation_MSE': mean_val_mse})
    sio.savemat(save_dir + f"/validation_acc_{lat_dim}.mat", mdict={'validation_acc': mean_val_acc})

    colors_data_points = plt.cm.gist_rainbow(np.linspace(0, 1, 103))
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.plot(latdim_range, mean_val_mse, color=colors_data_points[36])
    ax.set_xlabel("Latent dimension")
    ax.set_xlabel(f"Mean squared error ({k}-fold validation)")
    plt.show()
