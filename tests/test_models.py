import unittest
from cgmcore import modelutils
import os
import tensorflow as tf

setattr(tf.matmul, '__shallowcopy__', lambda self, _: self)


class TestModels(unittest.TestCase):

    def test_networks(self):
        """
        Tests all models.
        """

        input_shapes = [
            (32, 32, 32),
            (32, 32, 32),
            (32, 32, 32),
            (32, 32, 32),
            (30000, 3)
        ]
        output_size = 2

        creation_methods = [
            modelutils.create_dense_model,
            modelutils.create_voxnet_model_small,
            modelutils.create_voxnet_model_big,
            modelutils.create_voxnet_model_homepage,
            modelutils.create_point_net
        ]

        for input_shape, create_model in zip(input_shapes, creation_methods):
            model = create_model(input_shape, output_size)
            model_weights_path = "test.h5"
            model.save_weights(model_weights_path)
            model = create_model(input_shape, output_size)
            model.load_weights(model_weights_path)


if __name__ == '__main__':
    unittest.main()
