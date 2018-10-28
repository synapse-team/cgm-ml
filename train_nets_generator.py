from cgmcore import modelutils
from cgmcore import utils
import pickle
import numpy as np
import matplotlib.pyplot as plt
from keras import callbacks
import pprint
import glob
import pickle
import os
from cgmcore.etldatagenerator import get_dataset_path, create_datagenerator_from_parameters
import random

# Get the dataset path.
dataset_path = get_dataset_path()

# Hyperparameters.
steps_per_epoch = 100
validation_steps = 10
epochs = 100
batch_size = 32
random_seed = 667

# Create some generators.

# For creating pointclouds.
dataset_parameters_pointclouds = {}
dataset_parameters_pointclouds["input_type"] = "pointcloud"
dataset_parameters_pointclouds["output_targets"] = ["height"]
dataset_parameters_pointclouds["random_seed"] = random_seed
dataset_parameters_pointclouds["pointcloud_target_size"] = 10000
dataset_parameters_pointclouds["pointcloud_random_rotation"] = False
dataset_parameters_pointclouds["sequence_length"] = 0
datagenerator_instance_pointclouds = create_datagenerator_from_parameters(dataset_path, dataset_parameters_pointclouds)

# For creating voxelgrids.
dataset_parameters_voxelgrids = {}
dataset_parameters_voxelgrids["input_type"] = "voxelgrid"
dataset_parameters_voxelgrids["output_targets"] = ["height"]
dataset_parameters_voxelgrids["random_seed"] = random_seed
dataset_parameters_voxelgrids["voxelgrid_target_shape"] = (32, 32, 32)
dataset_parameters_voxelgrids["voxel_size_meters"] = 0.1
dataset_parameters_voxelgrids["voxelgrid_random_rotation"] = False
dataset_parameters_voxelgrids["sequence_length"] = 0
datagenerator_instance_voxelgrids = create_datagenerator_from_parameters(dataset_path, dataset_parameters_voxelgrids)

# Do the split.
assert np.array_equal(datagenerator_instance_pointclouds.qrcodes, datagenerator_instance_voxelgrids.qrcodes)
random.seed(random_seed)
qrcodes_shuffle = datagenerator_instance_pointclouds.qrcodes[:]
random.shuffle(qrcodes_shuffle)
split_index = int(0.8 * len(qrcodes_shuffle))
qrcodes_train = sorted(qrcodes_shuffle[:split_index])
qrcodes_validate = sorted(qrcodes_shuffle[split_index:])
del qrcodes_shuffle
print("QR-codes for training:\n", "\t".join(qrcodes_train))
print("QR-codes for validation:\n", "\t".join(qrcodes_validate))

# Create python generators.
generator_pointclouds_train = datagenerator_instance_pointclouds.generate(size=batch_size, qrcodes_to_use=qrcodes_train)
generator_pointclouds_validate = datagenerator_instance_pointclouds.generate(size=batch_size, qrcodes_to_use=qrcodes_validate)
generator_voxelgrids_train = datagenerator_instance_voxelgrids.generate(size=batch_size, qrcodes_to_use=qrcodes_train)
generator_voxelgrids_validate = datagenerator_instance_voxelgrids.generate(size=batch_size, qrcodes_to_use=qrcodes_validate)

# Testing the genrators.
def test_generator(generator):
    data = next(generator)
    print("Input:", data[0].shape, "Output:", data[1].shape)
test_generator(generator_pointclouds_train)
test_generator(generator_pointclouds_validate)
test_generator(generator_voxelgrids_train)
test_generator(generator_voxelgrids_validate)

# Output path. Ensure its existence.
output_path = "models"
if os.path.exists(output_path) == False:
    os.makedirs(output_path)
print("Using output path:", output_path)

# Important things.
pp = pprint.PrettyPrinter(indent=4)
tensorboard_callback = callbacks.TensorBoard()
histories = {}


# Method for saving model and history.
def save_model_and_history(model, history, name):

    print("Saving model and history...")

    datetime_string = utils.get_datetime_string()

    # Try to save model. Could fail.
    try:
        model_name = datetime_string + "-" + name + "-model.h5"
        model_path = os.path.join(output_path, model_name)
        model.save(model_path)
        print("Saved model to" + model_name)
    except Exception as e:
        print("WARNING! Failed to save model. Use model-weights instead.")

    # Save the model weights.
    model_weights_name = datetime_string + "-" + name + "-model-weights.h5"
    model_weights_path = os.path.join(output_path, model_weights_name)
    model.save_weights(model_weights_path)
    print("Saved model weights to" + model_name)

    # Save the history.
    history_name = datetime_string + "-" + name + "-history.p"
    history_path = os.path.join(output_path, history_name)
    pickle.dump(history.history, open(history_path, "wb"))
    print("Saved history to" + history_name)


# Training VoxNet.
def train_voxnet():
    print("Training VoxNet...")

    # Create the model.
    input_shape = (32, 32, 32)
    output_size = 1
    model_voxnet = modelutils.create_voxnet_model_homepage(input_shape, output_size)
    model_voxnet.summary()

    # Compile the model.
    model_voxnet.compile(
            optimizer="rmsprop",
            loss="mse",
            metrics=["mae"]
        )

    # Train the model.
    history = model_voxnet.fit_generator(
        generator_voxelgrids_train,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=generator_voxelgrids_validate,
        validation_steps=validation_steps,
        callbacks=[tensorboard_callback]
        )

    histories["voxnet"] = history
    save_model_and_history(model_voxnet, history, "voxnet")

    
# Training PointNet.
def train_pointnet():
    #dataset_name = dataset_name_pointcloud
    #print("Loading dataset...")
    #(x_input_train, y_output_train, _), (x_input_test, y_output_test, _), dataset_parameters = pickle.load(open(dataset_name, "rb"))
    #pp.pprint(dataset_parameters)

    #def transform(x_input, y_output):
    #    x_input_transformed = []
    #    y_output_transformed = []
    #    for input_sample, output_sample in zip(x_input_train, y_output_train):
    #        if input_sample.shape[0] == 30000:
    #            x_input_transformed.append(input_sample[:,0:3])
    #            y_output_transformed.append(output_sample)
    #        else:
    #            # TODO maybe do some padding here?
    #            print("Ignoring shape:", input_sample.shape)

    #    x_input_transformed = np.array(x_input_transformed)
    #    y_output_transformed = np.array(y_output_transformed)
    #    return x_input_transformed, y_output_transformed

    #x_input_train, y_output_train = transform(x_input_train, y_output_train)
    #x_input_test, y_output_test = transform(x_input_test, y_output_test)

    #print("Training data input shape:", x_input_train.shape)
    #print("Training data output shape:", y_output_train.shape)
    #print("Testing data input shape:", x_input_test.shape)
    #print("Testing data output shape:", y_output_test.shape)
    #print("")

    input_shape = (10000, 3)
    output_size = 1
    model_pointnet = modelutils.create_point_net(input_shape, output_size)
    model_pointnet.summary()

     # Compile the model.
    model_pointnet.compile(
            optimizer="rmsprop",
            loss="mse",
            metrics=["mae"]
        )

    # Train the model.
    history = model_pointnet.fit_generator(
        generator_pointclouds_train,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=generator_pointclouds_validate,
        validation_steps=validation_steps,
        callbacks=[tensorboard_callback]
        )

    histories["pointnet"] = history
    save_model_and_history(model_pointnet, history, "pointnet")

# Train the nets.
train_voxnet()
#train_pointnet()

#def plot_histories(histories, names):
#    for index, (history, name) in enumerate(histories.items()):
#        for key, data in history.items():
#            plt.plot(data, label=name + "-" + key)
#
#    fig_name = utils.get_datetime_string() + "-histories.png"
#    fig_path = os.path.join(output_path, fig_name)
#    plt.savefig(fig_path)
#    plt.show()
#    plt.close()

#plot_histories(histories, ["voxnet", "pointnet"])
