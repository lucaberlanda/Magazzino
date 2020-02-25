import numpy as np


def hinge_loss_single(feature_vector, label, theta, theta_0):
    """
    Finds the hinge loss on a single data point given specific classification
    parameters.

    Args:
        feature_vector - A numpy array describing the given data point.
        label - A real valued number, the correct classification of the data
            point.
        theta - A numpy array describing the linear classifier.
        theta_0 - A real valued number representing the offset parameter.


    Returns: A real number representing the hinge loss associated with the
    given data point and parameters.
    """
    loss = max(0, 1 - label * (np.dot(theta.T, feature_vector) + theta_0))
    return loss


def hinge_loss_full(feature_matrix, labels, theta, theta_0):
    """
    Finds the total hinge loss on a set of data given specific classification
    parameters.

    Args:
        feature_matrix - A numpy matrix describing the given data. Each row
            represents a single data point.
        labels - A numpy array where the kth element of the array is the
            correct classification of the kth row of the feature matrix.
        theta - A numpy array describing the linear classifier.
        theta_0 - A real valued number representing the offset parameter.


    Returns: A real number representing the hinge loss associated with the
    given dataset and parameters. This number should be the average hinge
    loss across all of the points in the feature matrix.
    """
    loss = np.average(np.maximum(0, 1 - labels * (np.dot(theta, feature_matrix.T) + theta_0)))
    return loss


def perceptron_single_step_update(feature_vector, label, current_theta, current_theta_0):
    """
    Properly updates the classification parameter, theta and theta_0, on a
    single step of the perceptron algorithm.

    Args:
        feature_vector - A numpy array describing a single data point.
        label - The correct classification of the feature vector.
        current_theta - The current theta being used by the perceptron
            algorithm before this update.
        current_theta_0 - The current theta_0 being used by the perceptron
            algorithm before this update.

    Returns: A tuple where the first element is a numpy array with the value of
    theta after the current update has completed and the second element is a
    real valued number with the value of theta_0 after the current updated has
    completed.
    """
    if label * (np.dot(current_theta.T, feature_vector) + current_theta_0) <= 0:
        current_theta = current_theta + feature_vector * label
        current_theta_0 = current_theta_0 + label

    return current_theta, current_theta_0


def perceptron(feature_matrix, labels, T):
    """
    Runs the full perceptron algorithm on a given set of data. Runs T
    iterations through the data set, there is no need to worry about
    stopping early.

    NOTE: Please use the previously implemented functions when applicable.
    Do not copy paste code from previous parts.

    NOTE: Iterate the data matrix by the orders returned by get_order(feature_matrix.shape[0])

    Args:
        feature_matrix -  A numpy matrix describing the given data. Each row
            represents a single data point.
        labels - A numpy array where the kth element of the array is the
            correct classification of the kth row of the feature matrix.
        T - An integer indicating how many times the perceptron algorithm
            should iterate through the feature matrix.

    Returns: A tuple where the first element is a numpy array with the value of
    theta, the linear classification parameter, after T iterations through the
    feature matrix and the second element is a real number with the value of
    theta_0, the offset classification parameter, after T iterations through
    the feature matrix.
    """
    theta = np.array([0 for i in np.arange(feature_matrix.shape[1])])
    theta_0 = 0
    for t in range(T):
        for i in np.arange(feature_matrix.shape[0]):  # get_order(feature_matrix.shape[0])
            theta, theta_0 = perceptron_single_step_update(feature_matrix[i], labels[i], theta, theta_0)

    return theta, theta_0


def average_perceptron(feature_matrix, labels, T):
    """
    Runs the average perceptron algorithm on a given set of data. Runs T
    iterations through the data set, there is no need to worry about
    stopping early.

    NOTE: Please use the previously implemented functions when applicable.
    Do not copy paste code from previous parts.

    NOTE: Iterate the data matrix by the orders returned by get_order(feature_matrix.shape[0])


    Args:
        feature_matrix -  A numpy matrix describing the given data. Each row
            represents a single data point.
        labels - A numpy array where the kth element of the array is the
            correct classification of the kth row of the feature matrix.
        T - An integer indicating how many times the perceptron algorithm
            should iterate through the feature matrix.

    Returns: A tuple where the first element is a numpy array with the value of
    the average theta, the linear classification parameter, found after T
    iterations through the feature matrix and the second element is a real
    number with the value of the average theta_0, the offset classification
    parameter, found after T iterations through the feature matrix.

    Hint: It is difficult to keep a running average; however, it is simple to
    find a sum and divide.
    """
    theta = np.array([0 for i in np.arange(feature_matrix.shape[1])])
    theta_sum = np.array([0 for i in np.arange(feature_matrix.shape[1])])
    theta_0 = 0.0
    theta_0_sum = 0.0
    n_data_pts = feature_matrix.shape[0]
    for t in range(T):
        for i in np.arange(n_data_pts):  # get_order(feature_matrix.shape[0])
            theta, theta_0 = perceptron_single_step_update(feature_matrix[i], labels[i], theta, theta_0)
            theta_sum = np.add(theta_sum, theta)
            theta_0_sum = np.add(theta_0_sum, theta_0)

    theta_avg = theta_sum / (T * n_data_pts)
    theta_0_avg = theta_0_sum / (T * n_data_pts)

    return theta_avg, theta_0_avg


ar = np.array([[1, 5, 1, 7, 1], [-1, -5, -1, -3, 1], [8, -9, 1, 0, 0]])
lab = np.array([1, 1, -1])
a, b = average_perceptron(ar, lab, 4)

print(a, b)