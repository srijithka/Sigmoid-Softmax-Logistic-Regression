import numpy as np

"""This script implements the functions for reading data.
"""

def load_data(filename):
    """Load a given txt file.

    Args:
        filename: A string.

    Returns:
        raw_data: An array of shape [n_samples, 256].
        labels : An array of shape [n_samples,].
        
    """
    data= np.load(filename)
    x= data['x']
    y= data['y']
    return x, y

def train_valid_split(raw_data, labels, split_index):
	"""Split the original training data into a new training dataset
	and a validation dataset.
	n_samples = n_train_samples + n_valid_samples

	Args:
		raw_data: An array of shape [n_samples, 256].
        labels : An array of shape [n_samples,].
		split_index: An integer.

	"""
	return raw_data[:split_index], raw_data[split_index:], labels[:split_index], labels[split_index:]

def prepare_X(raw_X):
    """Extract features from raw_X as required.

    Args:
        raw_X: An array of shape [n_samples, 256].

    Returns:
        X: An array of shape [n_samples, n_features].
    """
    raw_image = raw_X.reshape((-1, 16, 16))

    n = len(raw_image)

    feature1 = []

	# Feature 1: Measure of Symmetry
	### YOUR CODE HERE
    for x in raw_image:
        flip_x = np.fliplr(x)
        abs_diff = np.abs(x - flip_x)
        pixel_sum = np.sum(abs_diff)
        pixel_sum = pixel_sum / 256
        pixel_sum = -1 * pixel_sum
        feature1.append(pixel_sum)
	### END YOUR CODE
        
    feature2 = []

	# Feature 2: Measure of Intensity
	### YOUR CODE HERE
    for x in raw_image:
        sum_x = np.sum(x)
        sum_x = sum_x / 256
        feature2.append(sum_x)


	
	### END YOUR CODE

	# Feature 3: Bias Term. Always 1.
    feature3 = []
	### YOUR CODE HERE
    for x in raw_image:
         feature3.append(1)
	
	### END YOUR CODE

	# Stack features together in the following order.
    X = []
	# [Feature 3, Feature 1, Feature 2]
	### YOUR CODE HERE
    for i in range(0, len(feature3)):
        temp = []
        temp.append(feature3[i])
        temp.append(feature1[i])
        temp.append(feature2[i])
        X.append(temp)
    X = np.array(X)
	### END YOUR CODE
    return X

def prepare_y(raw_y):
    """
    Args:
        raw_y: An array of shape [n_samples,].
        
    Returns:
        y: An array of shape [n_samples,].
        idx:return idx for data label 1 and 2.
    """
    y = raw_y
    idx = np.where((raw_y==1) | (raw_y==2))
    y[np.where(raw_y==0)] = 0
    y[np.where(raw_y==1)] = 1
    y[np.where(raw_y==2)] = 2

    return y, idx




