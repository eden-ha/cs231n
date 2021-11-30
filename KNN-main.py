from data_utils import load_CIFAR10
import numpy as np
import matplotlib.pyplot as plt
from KNN import KNearestNeighbor


if __name__ == "__main__":
    # Load the raw CIFAR-10 data.
    cifar10_dir = 'datasets/cifar-10-batches-py'

    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir) # a magic function we provide

    # As a sanity check, we print out the size of the training and test data.
    print('Training data shape: ', X_train.shape)
    print('Training labels shape: ', y_train.shape)
    print('Test data shape: ', X_test.shape)
    print('Test labels shape: ', y_test.shape)

    # Visualize some examples from the data-set.
    # We show a few examples of training images from each class.
    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    num_classes = len(classes)
    samples_per_class = 7
    for y, cls in enumerate(classes):
        idxs = np.flatnonzero(y_train == y)
        idxs = np.random.choice(idxs, samples_per_class, replace=False)
        for i, idx in enumerate(idxs):
            plt_idx = i * num_classes + y + 1
            plt.subplot(samples_per_class, num_classes, plt_idx)
            plt.imshow(X_train[idx].astype('uint8'))
            plt.axis('off')
            if i == 0:
                plt.title(cls)
    plt.show()

    # Subsample the data for more efficient code execution in this exercise
    num_training = 5000
    mask = list(range(num_training))
    X_train = X_train[mask]
    y_train = y_train[mask]

    num_test = 500
    mask = list(range(num_test))
    X_test = X_test[mask]
    y_test = y_test[mask]

    # Reshape the image data into rows
    X_train = np.reshape(X_train, (X_train.shape[0], -1))
    X_test = np.reshape(X_test, (X_test.shape[0], -1))
    print(X_train.shape, X_test.shape)

    # Create a kNN classifier instance.
    # Remember that training a kNN classifier is a noop:
    # the Classifier simply remembers the data and does no further processing
    classifier = KNearestNeighbor()
    classifier.train(X_train, y_train)

    dists = classifier.compute_distances_two_loops(X_test)
    print(dists.shape)

    # We can visualize the distance matrix: each row is a single test example and
    # its distances to training examples
    plt.imshow(dists, interpolation='none')
    plt.show()