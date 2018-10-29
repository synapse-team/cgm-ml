import warnings
warnings.filterwarnings("ignore")
import unittest
from cgmcore.preprocesseddatagenerator import get_dataset_path, create_datagenerator_from_parameters
import time
import pickle


class TestPreprocessedDataGenerator(unittest.TestCase):

    @unittest.skip("demonstrating skipping")
    def test_pointcloud_measuring_time(self):
        dataset_path = get_dataset_path()
        print("Using dataset path:", dataset_path)

        dataset_parameters_pointclouds = {}
        dataset_parameters_pointclouds["input_type"] = "pointcloud"
        dataset_parameters_pointclouds["random_seed"] = 666
        dataset_parameters_pointclouds["pointcloud_target_size"] = 30000
        dataset_parameters_pointclouds["pointcloud_random_rotation"] = True
        data_generator = create_datagenerator_from_parameters(dataset_path, dataset_parameters_pointclouds)
        
        start_time = time.time()
        pointclouds_count = 0
        for qrcode, paths in data_generator.qrcodes_dictionary.items():
            for path in paths:
                with open(path, "rb") as file:
                    (pointcloud, targets) = pickle.load(file)
                    pointclouds_count += 1
                    del pointcloud
                    del targets
        elapsed_time = time.time() - start_time
        print("Loaded {} pointclouds in {} seconds".format(pointclouds_count, elapsed_time))
        
        
    #@unittest.skip("demonstrating skipping")
    def test_pointcloud_generation(self):
        dataset_path = get_dataset_path()
        print("Using dataset path:", dataset_path)

        dataset_parameters_pointclouds = {}
        dataset_parameters_pointclouds["input_type"] = "pointcloud"
        dataset_parameters_pointclouds["random_seed"] = 666
        dataset_parameters_pointclouds["pointcloud_target_size"] = 3000
        dataset_parameters_pointclouds["pointcloud_random_rotation"] = True
        data_generator = create_datagenerator_from_parameters(dataset_path, dataset_parameters_pointclouds)

        data_generator.analyze_files()

        dataset = next(data_generator.generate(size=100, verbose=True))
        assert dataset[0].shape == (100, 3000, 4), str(dataset[0].shape)
        assert dataset[1].shape == (100, 1), str(dataset[1].shape)
        
        
    def test_voxelgrid_generation(self):
        dataset_path = get_dataset_path()
        print("Using dataset path:", dataset_path)

        dataset_parameters_voxelgrids = {}
        dataset_parameters_voxelgrids["input_type"] = "voxelgrid"
        dataset_parameters_voxelgrids["random_seed"] = 666
        dataset_parameters_voxelgrids["voxelgrid_target_shape"] = (32, 32, 32)
        dataset_parameters_voxelgrids["voxel_size_meters"] = 0.1
        data_generator = create_datagenerator_from_parameters(dataset_path, dataset_parameters_voxelgrids)

        data_generator.analyze_files()

        dataset = next(data_generator.generate(size=100, verbose=True))
        assert dataset[0].shape == (100, 3000, 4), str(dataset[0].shape)
        assert dataset[1].shape == (100, 1), str(dataset[1].shape)


if __name__ == '__main__':
    unittest.main()
