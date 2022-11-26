import numpy as np
import tensorflow as tf


def generate_data(target_number=9,
                  mean_bag_length=10, var_bag_length=2, num_bag_train=250,
                  num_bag_test=50, seed=1):
    (train_images, train_labels), (test_images, test_labels) \
        = tf.keras.datasets.mnist.load_data()
    random_state = np.random.RandomState(seed)
    bags_list_train = []
    labels_list_train = []
    for i in range(num_bag_train):
        bag_length = max(np.int(random_state.normal(
            mean_bag_length, var_bag_length, 1)), 1)
        indices = random_state.randint(0, 60000, bag_length)
        labels_in_bag = train_labels[indices]
        labels_in_bag = labels_in_bag == target_number
        bags_list_train.append(train_images[indices])
        labels_list_train.append(labels_in_bag)
    bags_list_test = []
    labels_list_test = []
    for i in range(num_bag_test):
        bag_length = max(np.int(random_state.normal(
            mean_bag_length, var_bag_length, 1)), 1)
        indices = random_state.randint(0, 60000, bag_length)
        labels_in_bag = test_labels[indices]
        labels_in_bag = labels_in_bag == target_number
        bags_list_test.append(test_images[indices])
        labels_list_test.append(labels_in_bag)
    return zip(bags_list_train, labels_list_train), zip(bags_list_test,
                                                        labels_list_test)
