'''
Preprocesses datasets.

Currently this script loads all the PCDs files and stores the pointclouds contained therein on the harddrive. Stores the pointclouds together with the target(s). Uses pickle for that.
'''

import pickle
import os
from cgmcore.etldatagenerator import get_dataset_path, create_datagenerator_from_parameters
import datasetparameters
import shutil
import progressbar
import glob
import numpy as np
from pyntcloud import PyntCloud

# Get the dataset path.
dataset_path = get_dataset_path()
timestamp = dataset_path.split("/")[-1]
print("Dataset-path:", dataset_path)
print("Timestamp:", timestamp)

# Getting the qr-codes.
paths = glob.glob(os.path.join(dataset_path, "*"))
qrcodes = sorted([path.split("/")[-1] for path in paths])
print("Found {} QR-codes.".format(len(qrcodes)))   
    
def get_qrcodes_dictionary():
    all_pcd_paths = []
    all_jpg_paths = []
    qrcodes_dictionary = {}
    for qrcode in qrcodes:
        qrcodes_dictionary[qrcode] = []
        measurement_paths = glob.glob(os.path.join(dataset_path, qrcode, "*"))
        for measurement_path in measurement_paths:
            # Getting PCDs.
            pcd_paths = glob.glob(os.path.join(measurement_path, "pcd", "*.pcd"))
             
            # Getting JPGs.
            jpg_paths = glob.glob(os.path.join(measurement_path, "jpg", "*.jpg"))

            # Loading the targets.
            target_path = os.path.join(measurement_path, "target.txt")
            target_file = open(target_path, "r")
            targets = np.array([float(value) for value in target_file.read().split(",")])
            target_file.close()

            # Done.
            qrcodes_dictionary[qrcode].append((pcd_paths, jpg_paths, targets))
            all_pcd_paths.extend(pcd_paths)
            all_jpg_paths.extend(jpg_paths)
    return qrcodes_dictionary
qrcodes_dictionary = get_qrcodes_dictionary()
            
# Ensure path. That is a folder with the timestamp.
preprocessed_path = os.path.join("../data/preprocessed", timestamp)
print("Using path \"{}\" for preprocessing...".format(preprocessed_path))
if os.path.exists(preprocessed_path):
    print("WARNING! Path already exists. Removing...")                           
    shutil.rmtree(preprocessed_path)
os.mkdir(preprocessed_path)
    
# Do...
def load_pointcloud(pcd_path):
    pointcloud = PyntCloud.from_file(pcd_path).points.values
    return pointcloud
    
def preprocess():
    bar = progressbar.ProgressBar(max_value=len(qrcodes))
    for qrcode_index, qrcode in enumerate(qrcodes):
        bar.update(qrcode_index)
        qrcode_path = os.path.join(preprocessed_path, qrcode)
        os.mkdir(qrcode_path)
            
        index = 0
        for pcd_paths, jpg_paths, targets in qrcodes_dictionary[qrcode]:
            for pcd_path in pcd_paths:
                try:
                    pointcloud = load_pointcloud(pcd_path)
                    pickle_output_path = os.path.join(qrcode_path, "{}.p".format(index))
                    pickle.dump((pointcloud, targets), open(pickle_output_path, "wb"))
                    del pointcloud
                    index += 1
                except Exception as e:
                    print(e)
                    print("Skipped", pcd_path, "due to error.")
    bar.finish()
preprocess()