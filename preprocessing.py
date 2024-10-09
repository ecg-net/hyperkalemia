#!/usr/bin/env python
# coding: utf-8
# %%
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm
from utils.ecg_utils import (
    remove_baseline_wander,
    wavelet_denoise_signal,
    plot_12_lead_ecg,
)


# %%
def calculate_means_stds(npy_directory, n):
    
    npy_directory = Path(npy_directory)
    filelist = os.listdir(npy_directory)
    np.random.shuffle(filelist)

    full_batch = np.zeros((n, 5000, 12))
    count = 0

    for i, npy_filename in enumerate(tqdm(filelist[:n])):
        npy_filepath = npy_directory / npy_filename
        ekg_numpy_array = np.load(npy_filepath)

        if ekg_numpy_array.shape[0] != 5000:
            continue

        full_batch[count] = ekg_numpy_array
        count += 1

    full_batch = full_batch[:count]  # Trim the array to remove unused entries
    ecg_means = np.mean(full_batch, axis=(0, 1))
    ecg_stds = np.std(full_batch, axis=(0, 1))

    if ecg_means.shape[0] == ecg_stds.shape[0] == 12:
        print('Shape of mean and std for ECG normalization are correct!')

    return ecg_means, ecg_stds


# %%
# run the function on a list of filenames.
def ecg_denoising(
        raw_directory=raw_directory,
        output_directory=output_directory,
        ecg_means = ecg_means,
        ecg_stds = ecg_stds
    ):

    filelist = os.listdir(raw_directory)
    
    for i, filename in enumerate(tqdm(filelist[:n])):
        
        # Signal processing
        raw_directory = Path(raw_directory)
        ecg_filepath = raw_directory / filename
        ecg_numpy_array = np.load(ecg_filepath)
        # 1. Wandering baseline removal
        ecg_numpy_array = remove_baseline_wander(
            ecg_numpy_array, sampling_frequency=sampling_frequency
        )

        # Discrete wavelet transform denoising
        for lead in range(12):
            ecg_numpy_array[:, lead] = wavelet_denoise_signal(ecg_numpy_array[:, lead])

        # Lead-wise normalization with precomputed means and standard deviations
        ecg_numpy_array = (ecg_numpy_array - ecg_means) / ecg_stds

        np.save(output_directory / filename, ecg_numpy_array)
        
        return True


# %%
def segmentation(data_path, manifest_path, output_path, length, steps):
    manifest = pd.read_csv(manifest_path)

    data = []
    print('Staring segmenting ECG......')
    for index in tqdm(range(manifest.shape[0])):
        mrn = manifest['MRN'].iloc[index]            #MRN as column name for medical record number
        filename = manifest['filename'].iloc[index]  #filename as column name for ecg filename
        k = manifest['TEST_RSLT'].iloc[index]        #TEST_RSLT as column name for potassium level

        ecg_array = np.load(os.path.join(data_path, filename))
        ecg_array = ecg_array[:, 0]  # assume lead I is the first lead in npy file

        # Loop through every second as start point:
        for start in range(0, len(ecg_array), 500*steps):
            end = start + 500 * length    # 500 points for each seconds

            if start >= 0 and end <= len(ecg_array):
                sample = ecg_array[start:end]

                if len(sample) == 500 * length:
                    data.append({'mrn': mrn, 'original_filename': filename, 'ecg': sample, 'label': k})
                else:
                    print(f'Different sample size for {filename}: {len(sample)}')

    df = pd.DataFrame(data)
    df['filename'] = None

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    print('Saving segmented ECG......')
    for index, row in df.iterrows():
        original_filename = row['original_filename']
        ecg_array = row['ecg']
        new_file_name = f"{original_filename.rsplit('.', 1)[0]}_{index+1}.npy"
        new_file_path = os.path.join(output_path, new_file_name)
        
        df.at[index, 'filename'] = new_file_name
        np.save(new_file_path, ecg_array)

    df.drop(columns=['ecg'], inplace=True)
    df.to_csv(f'{output_path}/{length}seconds_length_{steps}seconds_step_ecg_manifest.csv')


# %%
if __name__ == "__main__":
    npy_directory = "ecg folder for entire database" 
    n = 100000  # the number of ecg for calculating mean and std
    
    print('Calculating ECG means and stds........')
    ecg_means, ecg_stds = calculate_means_stds(npy_directory, n)

    raw_directory = "path/to/raw_data_directory"  #raw ecg folder for target task
    output_directory = "path/to/output_directory" #output ecg folder for target task

    print('Denoising and Normalizing ECGs........')
    ecg_denoising(raw_directory, output_directory, ecg_means, ecg_stds)
    
    
    data_path = output_directory  # Output directory from the above step
    manifest_path = "/path/to/manifest.csv"  # Manifest file path
    output_path = "path/to/output_path"  # Output path for segmented ECGs
    length = 5  # Length of each segment in seconds
    steps = 1  # Number of seconds step

    process_ecg(data_path, manifest_path, output_path, length, steps)

