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
import multiprocessing as mp


def main():

    # Get the dataset path.
    global dataset_path
    global timestamp
    dataset_path = get_dataset_path()
    timestamp = dataset_path.split("/")[-1]
    print("Dataset-path:", dataset_path)
    print("Timestamp:", timestamp)


    # Ensure path for preprocessed data. That is a folder with the timestamp.
    global preprocessed_path
    preprocessed_path = os.path.join("../data/preprocessed", timestamp)
    print("Using path \"{}\" for preprocessing...".format(preprocessed_path))
    if os.path.exists(preprocessed_path):
        print("WARNING! Path already exists. Removing...")                           
        shutil.rmtree(preprocessed_path)
    os.mkdir(preprocessed_path)

    # Getting the qr-codes.
    paths = glob.glob(os.path.join(dataset_path, "*"))
    qrcodes = sorted([path.split("/")[-1] for path in paths])
    print("Found {} QR-codes.".format(len(qrcodes)))   

    # Getting the number of CPU-cores for multiprocessing.
    number_of_cores = mp.cpu_count()
    print("Number of CPUs-cores:", number_of_cores)
    number_of_processes = number_of_cores

    # Splitting QR-codes into subsets. One for each process.
    qrcodes_subsets = []
    split_size = 1 + len(qrcodes) // number_of_processes
    for process_index in range(number_of_processes):
        qrcodes_subset = qrcodes[(process_index) * split_size:(process_index + 1) * split_size]
        qrcodes_subsets.append(qrcodes_subset)

    # Double check the split.
    assert len(qrcodes) == np.sum([len(qrcodes_subset) for qrcodes_subset in qrcodes_subsets])
    assert np.array_equal(qrcodes, np.concatenate(qrcodes_subsets))

    # Create an output_queue.
    global output_queue
    output_queue = mp.Queue()

    # Create processes.
    processes = []
    for qrcodes_subset in qrcodes_subsets:
        process_target = process_qrcodes_subset_multiprocessing
        process_args = (qrcodes_subset,)
        process = mp.Process(target=process_target, args=process_args)
        processes.append(process)

    # Start the processes.
    for p in processes:
        p.start()

    # Exit the completed processes.
    for p in processes:
        p.join()

    # Get process results from the output queue
    output = [output_queue.get() for p in processes]
    print("Wrote {} files.".format(np.sum(output)))

    
def process_qrcodes_subset_multiprocessing(qrcodes_subset):
    qrcodes_dictionary = get_qrcodes_dictionary(qrcodes_subset)
    count = preprocess(qrcodes_subset, qrcodes_dictionary)
    output_queue.put(count)

    
def get_qrcodes_dictionary(qrcodes):
    all_pcd_paths = []
    all_jpg_paths = []
    qrcodes_dictionary = {}
    for qrcode in qrcodes:
        qrcodes_dictionary[qrcode] = []
        measurement_paths = glob.glob(os.path.join(dataset_path, qrcode, "*"))
        for measurement_path in measurement_paths:
            # Getting PCDs.
            pcd_paths = glob.glob(os.path.join(measurement_path, "pcd", "*.pcd"))
            if len(pcd_paths) == 0:
                continue
            
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
       
    
def preprocess(qrcodes, qrcodes_dictionary):
    count = 0
    bar = progressbar.ProgressBar(max_value=len(qrcodes))
    for qrcode_index, qrcode in enumerate(qrcodes):
        bar.update(qrcode_index)
        qrcode_path = os.path.join(preprocessed_path, qrcode)
        os.mkdir(qrcode_path)
            
        file_index = 0
        for pcd_paths, _, targets in qrcodes_dictionary[qrcode]:
            for pcd_path in pcd_paths:
                try:
                    pointcloud = load_pointcloud(pcd_path)
                    pickle_output_path = os.path.join(qrcode_path, "{}.p".format(file_index))
                    pickle.dump((pointcloud, targets), open(pickle_output_path, "wb"))
                    del pointcloud
                    file_index += 1
                    count += 1
                except Exception as e:
                    print(e)
                    print("Skipped", pcd_path, "due to error.")
    bar.finish()
    return count


def load_pointcloud(pcd_path):
    pointcloud = PyntCloud.from_file(pcd_path).points.values
    return pointcloud



if __name__ == "__main__":
    main()