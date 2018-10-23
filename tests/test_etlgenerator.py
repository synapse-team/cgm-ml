import warnings
warnings.filterwarnings("ignore")
import unittest
from cgmcore.etldatagenerator import get_dataset_path, create_datagenerator_from_parameters


class TestETLGenerator(unittest.TestCase):

    #@unittest.skip("demonstrating skipping")
    def test_pointcloud_generation(self):
        dataset_path = get_dataset_path()
        print("Using dataset path:", dataset_path)

        dataset_parameters_pointclouds = {}
        dataset_parameters_pointclouds["input_type"] = "pointcloud"
        dataset_parameters_pointclouds["output_targets"] = ["height", "weight"]
        dataset_parameters_pointclouds["random_seed"] = 666
        dataset_parameters_pointclouds["pointcloud_target_size"] = 30000
        dataset_parameters_pointclouds["pointcloud_random_rotation"] = True
        dataset_parameters_pointclouds["dataset_size_train"] = 1000
        dataset_parameters_pointclouds["dataset_size_test"] = 20

        data_generator = create_datagenerator_from_parameters(dataset_path, dataset_parameters_pointclouds)

        #print("\n".join(data_generator.qrcodes) + "\n{} QR-codes.".format(len(data_generator.qrcodes)))

        data_generator.analyze_files()

        #dataset = next(data_generator.generate(size=1, yield_file_paths=True, verbose=True))
        #assert dataset[0].shape == (1, 30000, 4)


if __name__ == '__main__':
    unittest.main()
