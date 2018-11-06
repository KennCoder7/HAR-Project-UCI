import numpy as np
import pandas as pd
from collections import Counter

DATASET_PATH = "./UCI_data/raw/UCI HAR Dataset/"


INPUT_SIGNAL_TYPES = [
    "body_acc_x_",
    "body_acc_y_",
    "body_acc_z_",
    "body_gyro_x_",
    "body_gyro_y_",
    "body_gyro_z_",
    "total_acc_x_",
    "total_acc_y_",
    "total_acc_z_"
]


def read_array(segmt, n):
    return segmt[:, n].reshape(segmt.shape[0])


def not_exist(arr, array):
    for i in range(len(array)):
        if arr[0] == array[i]:
            if i != len(array)-1:
                if arr[1] == array[i+1]:
                    return False
    return True


def permutation_algorithm(input_array):
    index = [1]
    output_array = [input_array[0]]
    i = 1
    j = i + 1
    while i != j:
        if j > len(input_array):
            j = 1
        elif not_exist([i, j], index) and not_exist([j, i], index):
            output_array.append(input_array[j - 1])
            index.append(j)
            i = j
            j = j + 1
        else:
            j = j + 1
    return output_array[0:-1]  # 123456789 135792468 147158259 369483726


def seg_permutation_algorithm(data):
    new_arr1 = permutation_algorithm(read_array(data, 0))
    for j in range(data.shape[1]):
        if j != 0:
            new_arr1 = np.vstack((new_arr1, permutation_algorithm(read_array(data, j))))
    return new_arr1.transpose().reshape((new_arr1.shape[1], new_arr1.shape[0], 1))  # (36, 128, 1)


def dataset_permutation_algorithm(data, proc_name="Permutation Algorithm"):
    new_data = seg_permutation_algorithm(data[0]).reshape((1, seg_permutation_algorithm(data[0]).shape[0],
                                                           seg_permutation_algorithm(data[0]).shape[1], 1))
    n = 0
    for i in range(data.shape[0]):
        if i != 0:
            new_data = np.vstack((new_data, seg_permutation_algorithm(data[i]).
                                  reshape((1, seg_permutation_algorithm(data[0]).shape[0],
                                           seg_permutation_algorithm(data[0]).shape[1], 1))))
        if i - n > 0.05 * data.shape[0]:
            n = i
            print("### Process --- (", proc_name, "_ data ) --- In progress --- [ ",
                  int(100 * round(i / data.shape[0], 2)), "% ] Finished ###")
    return new_data


def load_x(data_paths):
    X_signals = []

    for signal_type_path in data_paths:
        with open(signal_type_path, "r") as f:
            X_signals.append(
                [np.array(serie, dtype=np.float32)
                 for serie in [row.replace('  ', ' ').strip().split(' ') for row in f]]
            )

    return np.transpose(X_signals, (1, 2, 0))


def load_y(y_path):
    # Read dataset from disk, dealing with text file's syntax
    with open(y_path, "r") as f:
        y = np.array(
            [elem for elem in [
                row.replace('  ', ' ').strip().split(' ') for row in f
            ]],
            dtype=np.int32
        )

    y = y.reshape(-1, )
    # Substract 1 to each output class for friendly 0-based indexing
    return y - 1


train_x_signals_paths = [
    DATASET_PATH + "train/Inertial Signals/" + signal + "train.txt" for signal in INPUT_SIGNAL_TYPES
]
test_x_signals_paths = [
    DATASET_PATH + "test/Inertial Signals/" + signal + "test.txt" for signal in INPUT_SIGNAL_TYPES
]

train_y_path = DATASET_PATH + "train/y_train.txt"
test_y_path = DATASET_PATH + "test/y_test.txt"

train_x = load_x(train_x_signals_paths)
test_x = load_x(test_x_signals_paths)

train_y = load_y(train_y_path)
test_y = load_y(test_y_path)
train_y_matrix = np.asarray(pd.get_dummies(train_y), dtype=np.int8)
test_y_matrix = np.asarray(pd.get_dummies(test_y), dtype=np.int8)

print(train_x.shape, test_x.shape)
reshaped_train_x = train_x.reshape(len(train_x), 1, 128, 9)
reshaped_test_x = test_x.reshape(len(test_x), 1, 128, 9)
print(reshaped_train_x.shape, reshaped_test_x.shape)
train_x_transpose = reshaped_train_x.transpose((0, 3, 2, 1))
test_x_transpose = reshaped_test_x.transpose((0, 3, 2, 1))
print(train_x_transpose.shape, test_x_transpose.shape)
train_x_algorithm = dataset_permutation_algorithm(train_x_transpose)
test_x_algorithm = dataset_permutation_algorithm(test_x_transpose)
print(train_x_algorithm.shape, test_x_algorithm.shape)
print(train_y, Counter(train_y))
print(test_y, Counter(test_y))

print(train_y_matrix)

np.save("./UCI_data/processed/np_train_x.npy", train_x_algorithm)
np.save("./UCI_data/processed/np_train_y.npy", train_y_matrix)
np.save("./UCI_data/processed/np_test_x.npy", test_x_algorithm)
np.save("./UCI_data/processed/np_test_y.npy", test_y_matrix)


# (7352, 128, 9) (2947, 128, 9)
# (7352, 1, 128, 9) (2947, 1, 128, 9)
# (7352, 9, 128, 1) (2947, 9, 128, 1)
# ### Process --- ( Permutation Algorithm _ data ) Algorithm1 --- In progress --- [  5 % ] Finished ###
# ### Process --- ( Permutation Algorithm _ data ) Algorithm1 --- In progress --- [  10 % ] Finished ###
# ### Process --- ( Permutation Algorithm _ data ) Algorithm1 --- In progress --- [  15 % ] Finished ###
# ### Process --- ( Permutation Algorithm _ data ) Algorithm1 --- In progress --- [  20 % ] Finished ###
# ### Process --- ( Permutation Algorithm _ data ) Algorithm1 --- In progress --- [  25 % ] Finished ###
# ### Process --- ( Permutation Algorithm _ data ) Algorithm1 --- In progress --- [  30 % ] Finished ###
# ### Process --- ( Permutation Algorithm _ data ) Algorithm1 --- In progress --- [  35 % ] Finished ###
# ### Process --- ( Permutation Algorithm _ data ) Algorithm1 --- In progress --- [  40 % ] Finished ###
# ### Process --- ( Permutation Algorithm _ data ) Algorithm1 --- In progress --- [  45 % ] Finished ###
# ### Process --- ( Permutation Algorithm _ data ) Algorithm1 --- In progress --- [  50 % ] Finished ###
# ### Process --- ( Permutation Algorithm _ data ) Algorithm1 --- In progress --- [  55 % ] Finished ###
# ### Process --- ( Permutation Algorithm _ data ) Algorithm1 --- In progress --- [  60 % ] Finished ###
# ### Process --- ( Permutation Algorithm _ data ) Algorithm1 --- In progress --- [  65 % ] Finished ###
# ### Process --- ( Permutation Algorithm _ data ) Algorithm1 --- In progress --- [  70 % ] Finished ###
# ### Process --- ( Permutation Algorithm _ data ) Algorithm1 --- In progress --- [  75 % ] Finished ###
# ### Process --- ( Permutation Algorithm _ data ) Algorithm1 --- In progress --- [  80 % ] Finished ###
# ### Process --- ( Permutation Algorithm _ data ) Algorithm1 --- In progress --- [  85 % ] Finished ###
# ### Process --- ( Permutation Algorithm _ data ) Algorithm1 --- In progress --- [  90 % ] Finished ###
# ### Process --- ( Permutation Algorithm _ data ) Algorithm1 --- In progress --- [  95 % ] Finished ###
# ### Process --- ( Permutation Algorithm _ data ) Algorithm1 --- In progress --- [  5 % ] Finished ###
# ### Process --- ( Permutation Algorithm _ data ) Algorithm1 --- In progress --- [  10 % ] Finished ###
# ### Process --- ( Permutation Algorithm _ data ) Algorithm1 --- In progress --- [  15 % ] Finished ###
# ### Process --- ( Permutation Algorithm _ data ) Algorithm1 --- In progress --- [  20 % ] Finished ###
# ### Process --- ( Permutation Algorithm _ data ) Algorithm1 --- In progress --- [  25 % ] Finished ###
# ### Process --- ( Permutation Algorithm _ data ) Algorithm1 --- In progress --- [  30 % ] Finished ###
# ### Process --- ( Permutation Algorithm _ data ) Algorithm1 --- In progress --- [  35 % ] Finished ###
# ### Process --- ( Permutation Algorithm _ data ) Algorithm1 --- In progress --- [  40 % ] Finished ###
# ### Process --- ( Permutation Algorithm _ data ) Algorithm1 --- In progress --- [  45 % ] Finished ###
# ### Process --- ( Permutation Algorithm _ data ) Algorithm1 --- In progress --- [  50 % ] Finished ###
# ### Process --- ( Permutation Algorithm _ data ) Algorithm1 --- In progress --- [  55 % ] Finished ###
# ### Process --- ( Permutation Algorithm _ data ) Algorithm1 --- In progress --- [  60 % ] Finished ###
# ### Process --- ( Permutation Algorithm _ data ) Algorithm1 --- In progress --- [  65 % ] Finished ###
# ### Process --- ( Permutation Algorithm _ data ) Algorithm1 --- In progress --- [  70 % ] Finished ###
# ### Process --- ( Permutation Algorithm _ data ) Algorithm1 --- In progress --- [  75 % ] Finished ###
# ### Process --- ( Permutation Algorithm _ data ) Algorithm1 --- In progress --- [  80 % ] Finished ###
# ### Process --- ( Permutation Algorithm _ data ) Algorithm1 --- In progress --- [  85 % ] Finished ###
# ### Process --- ( Permutation Algorithm _ data ) Algorithm1 --- In progress --- [  90 % ] Finished ###
# ### Process --- ( Permutation Algorithm _ data ) Algorithm1 --- In progress --- [  95 % ] Finished ###
# (7352, 36, 128, 1) (2947, 36, 128, 1)
# [4 4 4 ... 1 1 1] Counter({5: 1407, 4: 1374, 3: 1286, 0: 1226, 1: 1073, 2: 986})
# [4 4 4 ... 1 1 1] Counter({5: 537, 4: 532, 0: 496, 3: 491, 1: 471, 2: 420})
# [[0 0 0 0 1 0]
#  [0 0 0 0 1 0]
#  [0 0 0 0 1 0]
#  ...
#  [0 1 0 0 0 0]
#  [0 1 0 0 0 0]
#  [0 1 0 0 0 0]]


