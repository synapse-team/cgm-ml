'''
This script trains on RGB-Maps.
'''
from cgmcore import modelutils
from cgmcore import utils
import numpy as np
from keras import models, layers, callbacks, optimizers
import pprint
import os
from cgmcore.preprocesseddatagenerator import get_dataset_path, create_datagenerator_from_parameters
import random
import qrcodes

# Get the dataset path.
dataset_path = get_dataset_path()
print("Using dataset path", dataset_path)

# Hyperparameters.
steps_per_epoch = 100
validation_steps = 10
epochs = 50
batch_size = 32
random_seed = 667

image_size = 128

# For creating pointclouds.
dataset_parameters_rgbmaps = {}
dataset_parameters_rgbmaps["input_type"] = "rgbmap"
dataset_parameters_rgbmaps["random_seed"] = 666
dataset_parameters_rgbmaps["filter"] = "360"
dataset_parameters_rgbmaps["sequence_length"] = 4
dataset_parameters_rgbmaps["rgbmap_target_width"] = image_size
dataset_parameters_rgbmaps["rgbmap_target_height"] = image_size
dataset_parameters_rgbmaps["rgbmap_scale_factor"] = 1.0
dataset_parameters_rgbmaps["rgbmap_axis"] = "horizontal"
datagenerator_instance = create_datagenerator_from_parameters(dataset_path, dataset_parameters_rgbmaps)

# Get the QR-codes.
qrcodes_to_use = datagenerator_instance.qrcodes[:]
#qrcodes_to_use = qrcodes.standing_list

# Do the split.
random.seed(random_seed)
qrcodes_shuffle = qrcodes_to_use[:]
random.shuffle(qrcodes_shuffle)
split_index = int(0.8 * len(qrcodes_shuffle))
qrcodes_train = sorted(qrcodes_shuffle[:split_index])
qrcodes_validate = sorted(qrcodes_shuffle[split_index:])
del qrcodes_shuffle
print("QR-codes for training:\n", "\t".join(qrcodes_train))
print("QR-codes for validation:\n", "\t".join(qrcodes_validate))

# Create python generators.
generator_train = datagenerator_instance.generate(size=batch_size, qrcodes_to_use=qrcodes_train)
generator_validate = datagenerator_instance.generate(size=batch_size, qrcodes_to_use=qrcodes_validate)

# Testing the genrators.
def test_generator(generator):
    data = next(generator)
    print("Input:", data[0].shape, "Output:", data[1].shape)
test_generator(generator_train)
test_generator(generator_validate)

# Training details.
training_details = {
    "dataset_path" : dataset_path,
    "qrcodes_train" : qrcodes_train,
    "qrcodes_validate" : qrcodes_validate,
    "steps_per_epoch" : steps_per_epoch,
    "validation_steps" : validation_steps,
    "epochs" : epochs,
    "batch_size" : batch_size,
    "random_seed" : random_seed,
}

# Output path. Ensure its existence.
output_path = "models"
if os.path.exists(output_path) == False:
    os.makedirs(output_path)
print("Using output path:", output_path)

# Important things.
pp = pprint.PrettyPrinter(indent=4)
tensorboard_callback = callbacks.TensorBoard()
histories = {}
    
# Training PointNet.
def train_rgbmaps():

    sequence_length = dataset_parameters_rgbmaps["sequence_length"]
    
    model = models.Sequential()
    model.add(layers.Permute((2, 3, 1, 4), input_shape=(sequence_length, image_size, image_size, 3)))
    model.add(layers.Reshape((image_size, image_size, sequence_length * 3)))
    model.add(layers.Conv2D(64, (3,3), activation="relu"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3,3), activation="relu"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3,3), activation="relu"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3,3), activation="relu"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.25))
    model.add(layers.Dense(128, activation="relu"))
    model.add(layers.Dropout(0.25))
    model.add(layers.Dense(1, activation="linear"))
    model.summary()
    
    
    # Compile the model.
    optimizer = "rmsprop"
    #optimizer = optimizers.Adagrad(lr=0.7, epsilon=None, decay=0.2)
    model.compile(
            optimizer=optimizer,
            loss="mse",
            metrics=["mae"]
        )

    # Train the model.
    history = model.fit_generator(
        generator_train,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=generator_validate,
        validation_steps=validation_steps,
        callbacks=[tensorboard_callback]
        )

    histories["pointnet"] = history
    modelutils.save_model_and_history(output_path, model, history, training_details, "pointnet")

train_rgbmaps()