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


# Finding the latest dataset-paths.
dataset_name_voxelgrid = utils.get_latest_preprocessed_dataset(path="datasets", filter="voxelgrid-dataset")
print(dataset_name_voxelgrid)
dataset_name_pointcloud = utils.get_latest_preprocessed_dataset(path="datasets", filter="pointcloud-dataset")
print(dataset_name_pointcloud)

# Output path. Ensure its existence.
output_path = "models"
if os.path.exists(output_path) == False:
    os.makedirs(output_path)
print("Using output path:", output_path)

# Important things.
pp = pprint.PrettyPrinter(indent=4)
tensorboard_callback = callbacks.TensorBoard()
histories = {}

# Hyperparameters.
epochs = 25

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
    dataset_name = dataset_name_voxelgrid
    print("Loading dataset...")
    (x_input_train, y_output_train, _), (x_input_test, y_output_test, _), dataset_parameters = pickle.load(open(dataset_name, "rb"))
    pp.pprint(dataset_parameters)
    print("Training data input shape:", x_input_train.shape)
    print("Training data output shape:", y_output_train.shape)
    print("Testing data input shape:", x_input_test.shape)
    print("Testing data output shape:", y_output_test.shape)
    print("")

    # Create the model.
    input_shape = (32, 32, 32)
    output_size = 2
    model_voxnet = modelutils.create_voxnet_model_homepage(input_shape, output_size)
    model_voxnet.summary()

    # Compile the model.
    model_voxnet.compile(
            optimizer="rmsprop",
            loss="mse",
            metrics=["mae"]
        )

    # Train the model.
    history = model_voxnet.fit(
        x_input_train, y_output_train,
        epochs=epochs,
        validation_data=(x_input_test, y_output_test),
        callbacks=[tensorboard_callback]
        )

    histories["voxnet"] = history
    save_model_and_history(model_voxnet, history, "voxnet")

# Training PointNet.
def train_pointnet():
    dataset_name = dataset_name_pointcloud
    print("Loading dataset...")
    (x_input_train, y_output_train, _), (x_input_test, y_output_test, _), dataset_parameters = pickle.load(open(dataset_name, "rb"))
    pp.pprint(dataset_parameters)

    def transform(x_input, y_output):
        x_input_transformed = []
        y_output_transformed = []
        for input_sample, output_sample in zip(x_input_train, y_output_train):
            if input_sample.shape[0] == 30000:
                x_input_transformed.append(input_sample[:,0:3])
                y_output_transformed.append(output_sample)
            else:
                # TODO maybe do some padding here?
                print("Ignoring shape:", input_sample.shape)

        x_input_transformed = np.array(x_input_transformed)
        y_output_transformed = np.array(y_output_transformed)
        return x_input_transformed, y_output_transformed

    x_input_train, y_output_train = transform(x_input_train, y_output_train)
    x_input_test, y_output_test = transform(x_input_test, y_output_test)

    print("Training data input shape:", x_input_train.shape)
    print("Training data output shape:", y_output_train.shape)
    print("Testing data input shape:", x_input_test.shape)
    print("Testing data output shape:", y_output_test.shape)
    print("")

    input_shape = (30000, 3)
    output_size = 2
    model_pointnet = modelutils.create_point_net(input_shape, output_size)
    model_pointnet.summary()

     # Compile the model.
    model_pointnet.compile(
            optimizer="rmsprop",
            loss="mse",
            metrics=["mae"]
        )

    # Train the model.
    history = model_pointnet.fit(
        x_input_train, y_output_train,
        epochs=epochs,
        validation_data=(x_input_test, y_output_test),
        callbacks=[tensorboard_callback],
        batch_size=4
        )

    histories["pointnet"] = history
    save_model_and_history(model_pointnet, history, "pointnet")

# Train the nets.
train_voxnet()
train_pointnet()

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
