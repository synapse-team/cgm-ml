#coding=utf-8

try:
    import sys
except:
    pass

try:
    import numpy as np
except:
    pass

try:
    import os
except:
    pass

try:
    import random
except:
    pass

try:
    from cgmcore.preprocesseddatagenerator import get_dataset_path
except:
    pass

try:
    from cgmcore.preprocesseddatagenerator import create_datagenerator_from_parameters
except:
    pass

try:
    from cgmcore import modelutils
except:
    pass

try:
    from keras import optimizers
except:
    pass

try:
    from hyperopt import Trials, STATUS_OK, tpe
except:
    pass

try:
    from hyperas import optim
except:
    pass

try:
    from hyperas.distributions import choice, uniform
except:
    pass
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperas.distributions import conditional

dataset_path = "../../preprocessed_trimmed/2018_07_31_10_52"
print("Using dataset path", dataset_path)
random_seed = 300
batch_size={{choice([15, 20, 25])}}
dataset_parameters_pointclouds = {}
dataset_parameters_pointclouds["input_type"] = "pointcloud"
dataset_parameters_pointclouds["output_targets"] = ["height"]
dataset_parameters_pointclouds["random_seed"] = random_seed
dataset_parameters_pointclouds["pointcloud_target_size"] = 10000
dataset_parameters_pointclouds["pointcloud_random_rotation"] = False
dataset_parameters_pointclouds["sequence_length"] = 0

datagenerator_instance_pointclouds = create_datagenerator_from_parameters(dataset_path, dataset_parameters_pointclouds)

# Get the QR-codes.
qrcodes_to_use = datagenerator_instance_pointclouds.qrcodes[0:1500]

# Do the split.
random.seed(random_seed)
qrcodes_shuffle = qrcodes_to_use[:]
random.shuffle(qrcodes_shuffle)
split_index_a = int(0.8 * len(qrcodes_shuffle))
#split_index_b = int(0.8 * 0.8 * len(qrcodes_shuffle))
qrcodes_train = sorted(qrcodes_shuffle[:split_index_a])
#qrcodes_validate = sorted(qrcodes_shuffle[split_index_b:split_index_a])
qrcodes_test = sorted(qrcodes_shuffle[split_index_a:])
del qrcodes_shuffle
#print("QR-codes for training:\n", "\t".join(qrcodes_train))
#print("QR-codes for validation:\n", "\t".join(qrcodes_validate))

# Create python generators.
generator_pointclouds_train = datagenerator_instance_pointclouds.generate(size=batch_size, qrcodes_to_use=qrcodes_train)
generator_pointclouds_test = datagenerator_instance_pointclouds.generate(size=batch_size, qrcodes_to_use=qrcodes_test)
# generator_pointclouds_validate = datagenerator_instance_pointclouds.generate(size=batch_size, qrcodes_to_use=qrcodes_validate)

size = 16000
generator_pointclouds = generator_pointclouds_train
X = []
Y = []
d = next(generator_pointclouds)
while d:
    t_x, t_y = d
    for x in t_x:
        X.append(x)
    for y in t_y:
        Y.append(y)
    d = next(generator_pointclouds)
    if len(X) > size:
        break

train_x = X
train_y = Y

size = 4000
generator_pointclouds = generator_pointclouds_test
X = []
Y = []
d = next(generator_pointclouds)
while d:
    t_x, t_y = d
    for x in t_x:
        X.append(x)
    for y in t_y:
        Y.append(y)
    d = next(generator_pointclouds)
    if len(X) > size:
        break

test_x = X
test_y = Y



def keras_fmin_fnct(space):

    """
    Creates a PointNet.

    See https://github.com/garyloveavocado/pointnet-keras/blob/master/train_cls.py

    Args:
        input_shape (shape): Input-shape.
        output_size (int): Output-size.

    Returns:
        Model: A model.
    """

    num_points = input_shape[0]

    def mat_mul(A, B):
        result = tf.matmul(A, B)
        return result

    input_points = layers.Input(shape=input_shape)
    x = layers.Convolution1D(64, 1, activation='relu',
                      input_shape=input_shape)(input_points)
    x = layers.BatchNormalization()(x)
    x = layers.Convolution1D(128, 1, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Convolution1D(1024, 1, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=num_points)(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(9, weights=[np.zeros([256, 9]), np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32)])(x)
    input_T = layers.Reshape((3, 3))(x)

    # forward net
    g = layers.Lambda(mat_mul, arguments={'B': input_T})(input_points)
    g = layers.Convolution1D(64, 1, input_shape=input_shape, activation='relu')(g)
    g = layers.BatchNormalization()(g)
    g = layers.Convolution1D(64, 1, input_shape=input_shape, activation='relu')(g)
    g = layers.BatchNormalization()(g)

    # feature transform net
    f = layers.Convolution1D(64, 1, activation='relu')(g)
    f = layers.BatchNormalization()(f)
    f = layers.Convolution1D(128, 1, activation='relu')(f)
    f = layers.BatchNormalization()(f)
    f = layers.Convolution1D(1024, 1, activation='relu')(f)
    f = layers.BatchNormalization()(f)
    f = layers.MaxPooling1D(pool_size=num_points)(f)
    f = layers.Dense(512, activation='relu')(f)
    f = layers.BatchNormalization()(f)
    f = layers.Dense(256, activation='relu')(f)
    f = layers.BatchNormalization()(f)
    f = layers.Dense(64 * 64, weights=[np.zeros([256, 64 * 64]), np.eye(64).flatten().astype(np.float32)])(f)
    feature_T = layers.Reshape((64, 64))(f)

    # forward net
    g = layers.Lambda(mat_mul, arguments={'B': feature_T})(g)
    g = layers.Convolution1D(64, 1, activation='relu')(g)
    g = layers.BatchNormalization()(g)
    g = layers.Convolution1D(128, 1, activation='relu')(g)
    g = layers.BatchNormalization()(g)
    g = layers.Convolution1D(1024, 1, activation='relu')(g)
    g = layers.BatchNormalization()(g)

    # global_feature
    global_feature = layers.MaxPooling1D(pool_size=num_points)(g)

    # point_net_cls
    c = global_feature
    for hidden_size in hidden_sizes:
        c = layers.Dense(hidden_size, activation='relu')(c)
        c = layers.BatchNormalization()(c)
        c = layers.Dropout(rate=0.7)(c)
    
    c = layers.Dense(output_size, activation='linear')(c)
    prediction = layers.Flatten()(c)

    model = models.Model(inputs=input_points, outputs=prediction)
    
    sgd = optimizers.SGD(lr=0.0001,decay=1e-6,momentum=0.5,nesterov=True)
    
    model.compile(optimizer=sgd, loss="mse", metrics=["mae"])
    
    result = model.fit(
        train_x,
        batch_size=batch_size,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_split=0.2
        )
    
    #get the highest validation accuracy of the training epochs
    validation_acc = np.amax(result.history['val_acc']) 
    print('Best validation acc of epoch:', validation_acc)
    return {'loss': -validation_acc, 'status': STATUS_OK, 'model': model}

def get_space():
    return {
    }
