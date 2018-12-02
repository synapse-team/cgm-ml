
# coding: utf-8

# # Train PointNet (https://arxiv.org/abs/1612.00593).
# 
# This notebook shows you how to use the PreprocessedDataGenerator in order to train PointNet.
# 
# The PreprocessedDataGenerator uses preprocessed-data instead of ETL-data. Wheras ETL-data comes mainly as PCD-files, preprocessed-data comes mainly as pointclouds stored as numpy-arrays. We identified PCD-loading as a bottleneck. 

# In[1]:


import sys
sys.path.insert(0, "..")

import numpy as np
import os
import random


# # Get the dataset path.
# 
# This snippet shows you how to get the lates preprocessed path.

# In[2]:


from cgmcore.preprocesseddatagenerator import get_dataset_path

dataset_path = "../../preprocessed_trimmed/2018_07_31_10_52"
print("Using dataset path", dataset_path)


# # Hyperparameters.

# In[3]:


steps_per_epoch = 10
validation_steps = 10
epochs = 8
batch_size = 30
random_seed = 300


# # Create data-generator.
# 
# The method create_datagenerator_from_parameters is a convencience method. It allows you to instantiate a generator from a specification-dictionary.

# In[4]:


from cgmcore.preprocesseddatagenerator import create_datagenerator_from_parameters

dataset_parameters_pointclouds = {}
dataset_parameters_pointclouds["input_type"] = "pointcloud"
dataset_parameters_pointclouds["output_targets"] = ["height"]
dataset_parameters_pointclouds["random_seed"] = random_seed
dataset_parameters_pointclouds["pointcloud_target_size"] = 10000
dataset_parameters_pointclouds["pointcloud_random_rotation"] = False
dataset_parameters_pointclouds["sequence_length"] = 0
datagenerator_instance_pointclouds = create_datagenerator_from_parameters(dataset_path, dataset_parameters_pointclouds)


# # Getting the QR-Codes and do a train-validate-split.
# 
# The data-generator is perfectly capable of retrieving all QR-codes from the dataset. This snipped shows how to do so and how to split the QR-codes into two sets: Train and validate.

# In[5]:


# Get the QR-codes.
qrcodes_to_use = datagenerator_instance_pointclouds.qrcodes[0:1500]

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


# # Creating python generators for training and validation.
# 
# Now both QR-codes lists can be used for creating the actual generators. One for training and one for validation.

# In[6]:


# Create python generators.
generator_pointclouds_train = datagenerator_instance_pointclouds.generate(size=batch_size, qrcodes_to_use=qrcodes_train)
generator_pointclouds_validate = datagenerator_instance_pointclouds.generate(size=batch_size, qrcodes_to_use=qrcodes_validate)


# # Using the generator to create data manually.
# 
# Of course you can use the generator to create data manually anytime.

# In[7]:


train_x, train_y = next(generator_pointclouds_train)
print("Input-shape:", train_x.shape)
print("Output-shape:", train_y.shape)


# # Training-details.
# 
# Training-details are a dictionary that gets stored in a file after training. It is supposed to contain information that is valuable. For example data that is relevant for training including the hyper-parameters. Intended to be used when comparing different models.

# In[8]:


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


# # Training PointNet.
# 
# The module modelutils contains methods for creating Neural Nets. The following code shows how to instantiate and train PointNet.

# In[9]:


from cgmcore import modelutils

input_shape = (dataset_parameters_pointclouds["pointcloud_target_size"], 3)
output_size = 1
model_pointnet = modelutils.create_point_net(input_shape, output_size, hidden_sizes = [64])
model_pointnet.summary()
    
model_pointnet.compile(
    optimizer="rmsprop",
    loss="mse",
    metrics=["mae"]
    )

history = model_pointnet.fit_generator(
    generator_pointclouds_train,
    steps_per_epoch=steps_per_epoch,
    epochs=epochs,
    validation_data=generator_pointclouds_validate,
    validation_steps=validation_steps
    )


# # Saving everything.
# 
# This saves the model, its history and the training-details to some output directory. The created artifacts can later be uses in order to compare different models.

# In[10]:


output_path = "."

modelutils.save_model_and_history(output_path, model_pointnet, history, training_details, "pointnet")

