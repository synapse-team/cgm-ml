from __future__ import absolute_import
import os
import numpy as np
import glob2 as glob
#import json
#import random
#import keras.preprocessing.image as image_preprocessing
#import progressbar
#from pyntcloud import PyntCloud
#import matplotlib.pyplot as plt
#import multiprocessing as mp
#import uuid
#import pickle
from . import utils


class ETLDataGenerator(object):
    """
    This class generates data for training.
    """

    def __init__(
        self,
        dataset_path,
        input_type,
        output_targets,
        sequence_length=0,
        image_target_shape=(160, 90),
        voxelgrid_target_shape=(32, 32, 32),
        voxel_size_meters=0.01,
        voxelgrid_random_rotation=False,
        pointcloud_target_size=32000,
        pointcloud_random_rotation=False
        ):
        """
        Initializes a DataGenerator.

        Args:
            dataset_path (string): Where the raw data is.
            input_type (string): Specifies how the input-data for the Neural Network looks like. Either 'image', 'pointcloud', 'voxgrid'.
            output_targets (list of strings): A list of targets for the Neural Network. For example *['height', 'weight']*.
            sequence_length (int): Specifies the lenght of the sequences. 0 would yield no sequence at all.
            image_target_shape (2D tuple of ints): Target shape of the images.
            voxelgrid_target_shape (3D tuple of ints): Target shape of the voxelgrids.
            voxel_size_meters (float): Size of the voxels. That is, edge length.
            voxelgrid_random_rotation (bool): If True voxelgrids will be rotated randomly.
            pointcloud_target_size (int): Target size of the pointclouds.
            pointcloud_random_rotation (bool): If True pointclouds will be rotated randomly.

        """

        # Preconditions.
        assert os.path.exists(dataset_path), "dataset_path must exist: " + str(dataset_path)
        assert isinstance(input_type, str), "input_type must be string: " + str(input_type)
        assert isinstance(output_targets, list), "output_targets must be list: " + str(output_targets)
        if input_type == "image":
            assert len(image_target_shape) == 2, "image_target_shape must be 2-dimensional: " + str(image_target_shape)
        if input_type == "voxelgrid":
            assert len(voxelgrid_target_shape) == 3, "voxelgrid_target_shape must be 3-dimensional: " + str(voxelgrid_target_shape)

        # Assign the instance-variables.
        self.dataset_path = dataset_path
        self.input_type = input_type
        self.output_targets = output_targets
        self.sequence_length = sequence_length
        self.image_target_shape = image_target_shape
        self.voxelgrid_target_shape = voxelgrid_target_shape
        self.voxel_size_meters = voxel_size_meters
        self.voxelgrid_random_rotation = voxelgrid_random_rotation
        self.pointcloud_target_size = pointcloud_target_size
        self.pointcloud_random_rotation = pointcloud_random_rotation

        # Find all QR-codes.
        self._find_qrcodes()
        assert self.qrcodes != [], "No QR-codes found!"

        self._prepare_data()

        return

        # HERE COMES OLD STUFF

        # Create some caches.
        self.image_cache = {}
        self.voxelgrid_cache = {}
        self.pointcloud_cache = {}

        # Get all the paths.
        self._get_paths()

        # Check if paths are fine.
        if self.input_type == "image":
            assert self.jpg_paths != []
        elif self.input_type == "voxelgrid" or self.input_type == "pointcloud":
            assert self.pcd_paths != []
        else:
            raise Exception("Unexpected: " + self.input_type)
        assert self.json_paths_personal != []
        assert self.json_paths_measures != []



        # Create the QR-codes dictionary.
        self._create_qrcodes_dictionary()


    def _find_qrcodes(self):
        """
        Finds all QR-codes.

        Each individual is represented via a unique QR-codes. This method extracts the set of QR-codes.
        """

        # Retrieve the QR-codes from the folders.
        paths = glob.glob(os.path.join(self.dataset_path, "*"))
        self.qrcodes = sorted([path.split("/")[-1] for path in paths])

    def _prepare_data(self):

        self.data = {}
        for qrcode in self.qrcodes:
            self.data[qrcode] = []
            #print("QR-code:", qrcode)
            measurement_paths = glob.glob(os.path.join(self.dataset_path, qrcode, "*"))
            for measurement_path in measurement_paths:
                # Getting PCDs.
                pcd_paths = glob.glob(os.path.join(measurement_path, "*.pcd"))

                # Getting JPGs.
                jpg_paths = glob.glob(os.path.join(measurement_path, "*.jpg"))

                # Loading the targets.
                target_path = os.path.join(measurement_path, "target.txt")
                target_file = open(target_path, "r")
                targets = np.array([float(value) for value in target_file.read().split(",")])
                target_file.close()

                # Done.
                #print(qrcode, pcd_paths, targets)
                self.data[qrcode].append((pcd_paths, jpg_paths, targets))


    def analyze_files(self):

        for qrcode in self.qrcodes:
            print("QR-code:", qrcode)
            for pcd_paths, jpg_paths, targets in self.data[qrcode]:
                print("{} PCD-files for targets {}".format(len(pcd_paths), targets))
                print("{} JPG-files for targets {}".format(len(pcd_paths), targets))

            #measurement_paths = glob.glob(os.path.join(self.dataset_path, qrcode, "*"))
            #for measurement_path in measurement_paths:
            #    print("  ", measurement_path.split("/")[-1])
            #    pcd_paths = glob.glob(os.path.join(measurement_path, "*.pcd"))
            #    print("    ", "{} PCDs".format(len(pcd_paths)))
        #print("Number of JPGs:", len(self.jpg_paths))
        #print("Number of PCDs:", len(self.pcd_paths))
        #print("Number of JSONs (personal):", len(self.json_paths_personal))
        p#rint("Number of JSONs (measures):", len(self.json_paths_measures))


def create_datagenerator_from_parameters(dataset_path, dataset_parameters):
    print("Creating data-generator...")
    datagenerator = ETLDataGenerator(
        dataset_path=dataset_path,
        input_type=dataset_parameters["input_type"],
        output_targets=dataset_parameters["output_targets"],
        sequence_length=dataset_parameters.get("sequence_length", 0),
        voxelgrid_target_shape=dataset_parameters.get("voxelgrid_target_shape", None),
        voxel_size_meters=dataset_parameters.get("voxel_size_meters", None),
        voxelgrid_random_rotation=dataset_parameters.get("voxelgrid_random_rotation", None),
        pointcloud_target_size=dataset_parameters.get("pointcloud_target_size", None),
        pointcloud_random_rotation=dataset_parameters.get("pointcloud_random_rotation", None)
    )
    #datagenerator.print_statistics()
    return datagenerator


def get_dataset_path():
    if os.path.exists("etldatasetpath.txt"):
        with open("etldatasetpath.txt", "r") as file:
            dataset_path = file.read().replace("\n", "")
    else:
        # Finding the latest.
        dataset_paths = glob.glob("../data/etl/*")
        dataset_path = sorted(dataset_paths)[-1]

    return dataset_path
