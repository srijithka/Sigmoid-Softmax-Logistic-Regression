import os
import matplotlib.pyplot as plt
from LogisticRegression import logistic_regression
from LRM import logistic_regression_multiclass
from DataReader import *

data_dir = "../data/"
train_filename = "training.npz"
test_filename = "test.npz"
    
def visualize_features(X, y):
    '''This function is used to plot a 2-D scatter plot of training features. 

    Args:
        X: An array of shape [n_samples, 2].
        y: An array of shape [n_samples,]. Only contains 1 or -1.

    Returns:
        No return. Save the plot to 'train_features.*' and include it
        in submission.
    '''
    ### YOUR CODE HERE
	
    positive_indices = y == 1
    negative_indices = y == -1

   
    plt.scatter(X[positive_indices, 0], X[positive_indices, 1], c='green', label='Class 1')


    plt.scatter(X[negative_indices, 0], X[negative_indices, 1], c='yellow', label='Class -1')


    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('2-D Scatter Plot of Training Features')

    plt.legend()
	
    plt.savefig('train_features.png')



    ### END YOUR CODE

def visualize_result(X, y, W):

    # Plot decision boundary
    plt.clf() 
    plt.plot(X[y==1,0],X[y==1,1],'or' ,markersize=3) 
    plt.plot(X[y==-1,0],X[y==-1,1],'ob' ,markersize=3) 
    plt.legend(['1','2'],loc="lower left", title="Classes") 
    plt.xlabel("Feature 1") 
    plt.ylabel("Feature 2") 
    symmetry = np.array([X[:,0].min(), X[:,0].max()]) 
    db = (-W[0] - W[1]*symmetry)/W[2] 
    plt.plot(symmetry,db,'--k') 
    plt.xlim([-1,0])
    plt.ylim([-1,0])
    plt.savefig("train_result_sigmoid.png")

    
    '''This function is used to plot the sigmoid model after training. 

        Args:
            X: An array of shape [n_samples, 2].
            y: An array of shape [n_samples,]. Only contains 1 or -1.
            W: An array of shape [n_features,].
        
        Returns:
            No return. Save the plot to 'train_result_sigmoid.*' and include it
            in submission.
        '''
	### YOUR CODE HERE

	

	### END YOUR CODE
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def visualize_result_multi(X, y, W):
     ### YOUR CODE HERE
	x_cord = X[:, 0]
	y_cord = X[:, 1]
	plt.clf()
	plt.scatter(x_cord[y == 0], y_cord[y == 0], c='b', marker='o', label='0')
	plt.scatter(x_cord[y == 1], y_cord[y == 1], c='r', marker='x', label='1')
	plt.scatter(x_cord[y == 2], y_cord[y == 2], c='y', marker='v', label='2')
	x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
	y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
	xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
	# Z = np.dot(np.c_[xx.ravel(), yy.ravel()], W.T)
	Z = np.dot(np.c_[np.ones(xx.ravel().shape), xx.ravel(), yy.ravel()], W)
	Z = np.argmax(Z, axis=1)
	Z = Z.reshape(xx.shape)
	plt.contourf(xx, yy, Z, alpha=0.4)
	plt.xlabel('x axis')
	plt.ylabel('y axis')
	plt.legend()
	plt.savefig('train_result_softmax.png')

	### YOUR CODE HERE
     
     
       

	### END YOUR CODE

def main():
	# ------------Data Preprocessing------------
	# Read data for training.
    
    raw_data, labels = load_data(os.path.join(data_dir, train_filename))
    raw_train, raw_valid, label_train, label_valid = train_valid_split(raw_data, labels, 2300)

    ##### Preprocess raw data to extract features
    train_X_all = prepare_X(raw_train)
    valid_X_all = prepare_X(raw_valid)
    ##### Preprocess labels for all data to 0,1,2 and return the idx for data from '1' and '2' class.
    train_y_all, train_idx = prepare_y(label_train)
    valid_y_all, val_idx = prepare_y(label_valid)  

    ####### For binary case, only use data from '1' and '2'  
    train_X = train_X_all[train_idx]
    train_y = train_y_all[train_idx]
    ####### Only use the first 1350 data examples for binary training. 
    train_X = train_X[0:1350]
    train_y = train_y[0:1350]
    valid_X = valid_X_all[val_idx]
    valid_y = valid_y_all[val_idx]
    ####### set lables to  1 and -1. Here convert label '2' to '-1' which means we treat data '1' as postitive class. 
    train_y[np.where(train_y==2)] = -1
    valid_y[np.where(valid_y==2)] = -1
    data_shape= train_y.shape[0] 

#    # Visualize training data.
    visualize_features(train_X[:, 1:3], train_y)


   # ------------Logistic Regression Sigmoid Case------------

   ##### Check BGD, SGD, miniBGD
    logisticR_classifier = logistic_regression(learning_rate=0.5, max_iter=100)

    logisticR_classifier.fit_BGD(train_X, train_y)
    print(logisticR_classifier.get_params())
    print(logisticR_classifier.score(train_X, train_y))

    logisticR_classifier.fit_miniBGD(train_X, train_y, data_shape)
    print(logisticR_classifier.get_params())
    print(logisticR_classifier.score(train_X, train_y))

    logisticR_classifier.fit_SGD(train_X, train_y)
    print(logisticR_classifier.get_params())
    print(logisticR_classifier.score(train_X, train_y))

    logisticR_classifier.fit_miniBGD(train_X, train_y, 1)
    print(logisticR_classifier.get_params())
    print(logisticR_classifier.score(train_X, train_y))

    logisticR_classifier.fit_miniBGD(train_X, train_y, 10)
    print(logisticR_classifier.get_params())
    print(logisticR_classifier.score(train_X, train_y))


    # Explore different hyper-parameters.
    ### YOUR CODE HERE
	
    logisticR_classifier_learning = logistic_regression(learning_rate=0.1, max_iter=100)
    logisticR_classifier_learning.fit_miniBGD(train_X, train_y, 10)
    print('lower learning rate for mini batch of 10')
    print(logisticR_classifier_learning.get_params())
    print(logisticR_classifier_learning.score(train_X, train_y))
	
    logisticR_classifier_itr = logistic_regression(learning_rate=0.5, max_iter=500)
    logisticR_classifier_itr.fit_miniBGD(train_X, train_y, 10)
    print(' Increaing number of iterations to 500')
    print(logisticR_classifier_itr.get_params())
    print(logisticR_classifier_itr.score(train_X, train_y))
	
    logisticR_classifier_itr = logistic_regression(learning_rate=0.1, max_iter=500)
    logisticR_classifier_itr.fit_miniBGD(train_X, train_y, 10)
    print(' Increaing number of iterations to 500, low learning rate')
    print(logisticR_classifier_itr.get_params())
    print(logisticR_classifier_itr.score(train_X, train_y))
	

    ### END YOUR CODE

	# Visualize the your 'best' model after training.
	
	# visualize_result(train_X[:, 1:3], train_y, best_logisticR.get_params())
    best_logistic_classifier = logisticR_classifier
    visualize_result(train_X[:, 1:3], train_y, best_logistic_classifier.get_params())

    ### YOUR CODE HERE

    ### END YOUR CODE

    # Use the 'best' model above to do testing. Note that the test data should be loaded and processed in the same way as the training data.
    ### YOUR CODE HERE

    raw_test, label_test = load_data(os.path.join(data_dir, test_filename))
    test_X_all = prepare_X(raw_test) 
    test_y_all, test_idx = prepare_y(label_test)
    test_X = test_X_all[test_idx] 
    test_y = test_y_all[test_idx]
    test_y[np.where(test_y==2)] = -1 
    print("Test Data Output with params of best classifier according to my hyperparams")
    print(best_logistic_classifier.get_params())
    print(best_logistic_classifier.score(test_X, test_y))

    ### END YOUR CODE


    # ------------Logistic Regression Multiple-class case, let k= 3------------
    ###### Use all data from '0' '1' '2' for training
    train_X = train_X_all
    train_y = train_y_all
    valid_X = valid_X_all
    valid_y = valid_y_all

    #########  miniBGD for multiclass Logistic Regression
    logisticR_classifier_multiclass_best = logistic_regression_multiclass(learning_rate=0.5, max_iter=100,  k= 3)
    logisticR_classifier_multiclass_best.fit_miniBGD(train_X, train_y, 10)
    print(logisticR_classifier_multiclass_best.get_params())
    print(logisticR_classifier_multiclass_best.score(train_X, train_y))

    # Explore different hyper-parameters.
    ### YOUR CODE HERE

    logisticR_classifier_multiclass = logistic_regression_multiclass(learning_rate=0.1, max_iter=100,  k= 3)
    logisticR_classifier_multiclass.fit_miniBGD(train_X, train_y, 10)
    print("Multiclass with 0.1 learning rate and 100 iterations")
    print(logisticR_classifier_multiclass.get_params())
    print(logisticR_classifier_multiclass.score(train_X, train_y))

    logisticR_classifier_multiclass = logistic_regression_multiclass(learning_rate=0.1, max_iter=100,  k= 3)
    logisticR_classifier_multiclass.fit_miniBGD(train_X, train_y, 100)
    print("Multiclass with 0.1 learning rate and 100 iterations, 100 batch size")
    print(logisticR_classifier_multiclass.get_params())
    print(logisticR_classifier_multiclass.score(train_X, train_y))

    logisticR_classifier_multiclass = logistic_regression_multiclass(learning_rate=1, max_iter=100,  k= 3)
    logisticR_classifier_multiclass.fit_miniBGD(train_X, train_y, 10)
    print("Multiclass with 1 learning rate and 100 iterations, 10 batch size")
    print(logisticR_classifier_multiclass.get_params())
    print(logisticR_classifier_multiclass.score(train_X, train_y))

    logisticR_classifier_multiclass = logistic_regression_multiclass(learning_rate=1, max_iter=200,  k= 3)
    logisticR_classifier_multiclass.fit_miniBGD(train_X, train_y, 100)
    print("Multiclass with 1 learning rate and 200 iterations, 100 batch size")
    print(logisticR_classifier_multiclass.get_params())
    print(logisticR_classifier_multiclass.score(train_X, train_y))

    ### END YOUR CODE

	# Visualize the your 'best' model after training.
    visualize_result_multi(train_X[:, 1:3], train_y, logisticR_classifier_multiclass_best.get_params())


    # Use the 'best' model above to do testing.
    ### YOUR CODE HERE
    print("Testing accuracy for Test Data with my best LRM")
    print(logisticR_classifier_multiclass_best.score(test_X_all, test_y_all))

    ### END YOUR CODE


    # ------------Connection between sigmoid and softmax------------
    ############ Now set k=2, only use data from '1' and '2' 

    #####  set labels to 0,1 for softmax classifer
    train_X = train_X_all[train_idx]
    train_y = train_y_all[train_idx]
    train_X = train_X[0:1350]
    train_y = train_y[0:1350]
    valid_X = valid_X_all[val_idx]
    valid_y = valid_y_all[val_idx] 
    train_y[np.where(train_y==2)] = 0
    valid_y[np.where(valid_y==2)] = 0  
    
    ###### First, fit softmax classifer until convergence, and evaluate 
    ##### Hint: we suggest to set the convergence condition as "np.linalg.norm(gradients*1./batch_size) < 0.0005" or max_iter=10000:
    ### YOUR CODE HERE
    logisticR_classifier_multiclass = logistic_regression_multiclass(learning_rate=0.5, max_iter=10000,k= 2)
    logisticR_classifier_multiclass.fit_miniBGD(train_X, train_y, 10)
    print("\n\n2-Class Softmax LR")
    test_X = test_X_all[test_idx] 
    test_y = test_y_all[test_idx]
    test_y[np.where(test_y==2)] = 0
    print("The test accuracy for class 1 & 2 using LRM")
    print(logisticR_classifier_multiclass.score(test_X, test_y))

    ### END YOUR CODE






    train_X = train_X_all[train_idx]
    train_y = train_y_all[train_idx]
    train_X = train_X[0:1350]
    train_y = train_y[0:1350]
    valid_X = valid_X_all[val_idx]
    valid_y = valid_y_all[val_idx] 
    #####       set lables to -1 and 1 for sigmoid classifer
    train_y[np.where(train_y==2)] = -1
    valid_y[np.where(valid_y==2)] = -1   

    ###### Next, fit sigmoid classifer until convergence, and evaluate
    ##### Hint: we suggest to set the convergence condition as "np.linalg.norm(gradients*1./batch_size) < 0.0005" or max_iter=10000:
    ### YOUR CODE HERE
    best_logisticR = logistic_regression(learning_rate=0.5, max_iter=10000) 
    best_logisticR.fit_miniBGD(train_X, train_y,10)
    print("Sigmoid Classification with Convergance")
    test_X = test_X_all[test_idx] 
    test_y = test_y_all[test_idx] 
    test_y[np.where(test_y==2)] = -1 
    print("test accuracy for class 1 and 2 using Sigmoid for Convergence: ", best_logisticR.score(test_X,test_y))
    ### END YOUR CODE


    ################Compare and report the observations/prediction accuracy


    '''
    Explore the training of these two classifiers and monitor the graidents/weights for each step. 
    Hint: First, set two learning rates the same, check the graidents/weights for the first batch in the first epoch. What are the relationships between these two models? 
    Then, for what leaning rates, we can obtain w_1-w_2= w for all training steps so that these two models are equivalent for each training step. 
    '''
    ### YOUR CODE HERE

    # Setting same learning rate

    print("Setting same learninf rate")
    
    sig = logistic_regression(learning_rate=1, max_iter=1) 
    sig.fit_miniBGD(train_X, train_y, 10)
    print("Weights For learning rate of 1 for sigmoid : ")
    print(sig.get_params())
    train_y[np.where(train_y==-1)] = 0
    softmax = logistic_regression_multiclass(learning_rate=1, max_iter=1,k= 2)
    softmax.fit_miniBGD(train_X, train_y, 10)
    print("Weightas for Learning 1 for softmax :")
    print(softmax.get_params())

    print("Provind W' = w1 - w2")

    sig = logistic_regression(learning_rate=1, max_iter=1) 
    sig.fit_miniBGD(train_X, train_y, 10)
    print("Weights For learning rate of 1 for sigmoid : ")
    print(sig.get_params())
    train_y[np.where(train_y==-1)] = 0
    softmax = logistic_regression_multiclass(learning_rate=0.5, max_iter=1,k= 2)
    softmax.fit_miniBGD(train_X, train_y, 10)
    print("Weightas for Learning 0.5 for softmax :")
    print(softmax.get_params())
    ### END YOUR CODE

    # ------------End------------
    
    


    

if __name__ == '__main__':
	main()
    
    
