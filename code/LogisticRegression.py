import numpy as np
import sys

"""This script implements a two-class logistic regression model.
"""

class logistic_regression(object):
	
    def __init__(self, learning_rate, max_iter):
        self.learning_rate = learning_rate
        self.max_iter = max_iter

    def fit_BGD(self, X, y):
        """Train perceptron model on data (X,y) with Batch Gradient Descent.

        Args:
            X: An array of shape [n_samples, n_features].
            y: An array of shape [n_samples,]. Only contains 1 or -1.

        Returns:
            self: Returns an instance of self.
        """
        n_samples, n_features = X.shape

		### YOUR CODE HERE
        weights = np.random.random(n_features)
        self = self.assign_weights(weights)

        for it in range(0, self.max_iter):
            intial_gradient = np.zeros((n_features,))
            for i in range(0, n_samples):
                single_gradient = self._gradient(X[i], y[i])
                intial_gradient += single_gradient
            average_gradient = intial_gradient / n_samples
            self.W -= self.learning_rate * average_gradient

		### END YOUR CODE
        return self

    def fit_miniBGD(self, X, y, batch_size):
        """Train perceptron model on data (X,y) with mini-Batch Gradient Descent.

        Args:
            X: An array of shape [n_samples, n_features].
            y: An array of shape [n_samples,]. Only contains 1 or -1.
            batch_size: An integer.

        Returns:
            self: Returns an instance of self.
        """
		### YOUR CODE HERE
        n_samples, n_features = X.shape
        weights = np.random.random(n_features)
        self = self.assign_weights(weights)

        for it in range(0, self.max_iter):
            shuffled_indices = np.random.permutation(n_samples)
            X_shuffled = X[shuffled_indices]
            y_shuffled = y[shuffled_indices]

            for i in range(0, n_samples, batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]

                gradients = np.zeros_like(self.W)
                for xi, yi in zip(X_batch, y_batch):
                    single_gradient = self._gradient(xi, yi)
                    gradients += single_gradient
                average_gradient = gradients / len(X_batch)
                if np.linalg.norm(average_gradient) < 0.0005:
                    print("The number of iteration where the convergance occurs for sigmoid  is ", it)
                    print("Weight matrix for sigmoid is ", self.W)
                    return self
                self.W -= self.learning_rate * average_gradient
            

		### END YOUR CODE
        return self

    def fit_SGD(self, X, y):
        """Train perceptron model on data (X,y) with Stochastic Gradient Descent.

        Args:
            X: An array of shape [n_samples, n_features].
            y: An array of shape [n_samples,]. Only contains 1 or -1.

        Returns:
            self: Returns an instance of self.
        """
		### YOUR CODE HERE

        return self.fit_miniBGD(X, y, 1)

		### END YOUR CODE
        return self

    def _gradient(self, _x, _y):
        """Compute the gradient of cross-entropy with respect to self.W
        for one training sample (_x, _y). This function is used in fit_*.

        Args:
            _x: An array of shape [n_features,].
            _y: An integer. 1 or -1.

        Returns:
            _g: An array of shape [n_features,]. The gradient of
                cross-entropy with respect to self.W.
        """
		### YOUR CODE HERE
        sigm = np.dot(self.W, _x)
        sigm = sigm * _y
        exp = np.exp(-1 * sigm)
        gradient = []
        for i in _x:
            val = -1 * i * _y * exp
            val = val / (1 + exp)
            gradient.append(val)
        gradient = np.array(gradient)
        return gradient


		### END YOUR CODE

    def get_params(self):
        """Get parameters for this perceptron model.

        Returns:
            W: An array of shape [n_features,].
        """
        if self.W is None:
            print("Run fit first!")
            sys.exit(-1)
        return self.W

    def predict_proba(self, X):
        """Predict class probabilities for samples in X.

        Args:
            X: An array of shape [n_samples, n_features].

        Returns:
            preds_proba: An array of shape [n_samples, 2].
                Only contains floats between [0,1].
        """
		### YOUR CODE HERE

        sigm = np.dot(X, self.W)
        exp = np.exp(-1 * sigm)

        result_array = 1 / (1 + exp)
        result_2d_array = np.column_stack((result_array, 1 - result_array))

        return result_2d_array

		### END YOUR CODE


    def predict(self, X):
        """Predict class labels for samples in X.

        Args:
            X: An array of shape [n_samples, n_features].

        Returns:
            preds: An array of shape [n_samples,]. Only contains 1 or -1.
        """
		### YOUR CODE HERE

        predict_probablities = self.predict_proba(X)


        new_array = np.where(predict_probablities[:, 0] >= 0.5, 1, -1)

        return new_array
		### END YOUR CODE

    def score(self, X, y):
        """Returns the mean accuracy on the given test data and labels.

        Args:
            X: An array of shape [n_samples, n_features].
            y: An array of shape [n_samples,]. Only contains 1 or -1.

        Returns:
            score: An float. Mean accuracy of self.predict(X) wrt. y.
        """
		### YOUR CODE HERE

        n_samples = len(y)

        predict_probs = self.predict(X)

        ct = 0

        for i in range(0, n_samples):
            if predict_probs[i] == y[i]:
                ct += 1
        
        return ct / n_samples

		### END YOUR CODE
    
    def assign_weights(self, weights):
        self.W = weights
        return self

