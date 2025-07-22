import idx2numpy
import numpy as np
from sklearn.model_selection import train_test_split


def prepare_MNIST_data_sets(MNIST_data_folder, max_pixel_value=255, vectorize_features=True, display_details=False):
    """
    Preprocess the MNIST data into a train, validation and test data set.

    :param MNIST_data_folder: Local folder containing the MNIST data
    :param max_pixel_value: The maximum pixel value (used to standardize the regressor matrix).
    :param vectorize_features: Create vector stack of all input pixels
    :param display_details: Print details
    :return X_train: Regression matrix for training
    :return X_valid: Regression matrix for validation
    :return X_test: Regression matrix for testing
    :return y_train_digits: Digit in train data set (values in 0, 1, ..., 9)
    :return y_valid_digits: Digit in validation data set (values in 0, 1, ..., 9)
    :return y_test_digits: Digit in test data set (values in 0, 1, ..., 9)
    """

    # Read all data
    train_images = idx2numpy.convert_from_file(MNIST_data_folder+"train-images.idx3-ubyte")/max_pixel_value
    train_labels = idx2numpy.convert_from_file(MNIST_data_folder+"train-labels.idx1-ubyte")
    test_images = idx2numpy.convert_from_file(MNIST_data_folder+"t10k-images.idx3-ubyte")/max_pixel_value
    test_labels = idx2numpy.convert_from_file(MNIST_data_folder+"t10k-labels.idx1-ubyte")

    # Dimensions
    train_sample_size = train_images.shape[0]
    test_sample_size = test_images.shape[0]
    image_width = train_images.shape[1]
    image_height = train_images.shape[2]

    # The original train data will be split into a train data set (50k) and validation data set (10k)
    if display_details:
        print("Dimensions of input data:")
        print(f"Train: {train_sample_size}")
        print(f"Test: {test_sample_size}")

    # Split train data (stratify by image digit to preserve distribution among classes)
    if vectorize_features:
        X_full = train_images.reshape(train_sample_size, image_width*image_height)
        X_test = test_images.reshape(test_sample_size, image_width*image_height)
    else:
        X_full = train_images
        X_test = test_images
    
    # Split train data (stratify by image digit to preserve distribution among classes)
    X_train, X_valid, y_train_digits, y_valid_digits = train_test_split(X_full, train_labels, test_size = 1/6, random_state=42, stratify=train_labels)

    # Outputs for test set
    y_test_digits = test_labels

    return X_train, X_valid, X_test, y_train_digits, y_valid_digits, y_test_digits


def digit_label_statistics(y_digits, output_decimals=2):
    """
    Calculate sample size and digit distribution from input vector containing digits.

    :param y_digits: Numpy vector with digits
    :param output_decimals: Decimals to keep in output
    :return overview: Numpy array containing [sample size, percentage 0, percentage 1, ..., percentage 9]
    
    """

    # Sample size
    sample_size = y_digits.shape[0]

    # Count digit labels
    _, digit_counts = np.unique(y_digits, return_counts=True)
    digit_percentages= np.round(digit_counts/sample_size*100, decimals= output_decimals)

    overview = np.insert(digit_percentages, 0, sample_size)

    return overview







