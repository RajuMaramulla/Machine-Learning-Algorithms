#Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Download the dataset from Kaggle
#https://www.kaggle.com/andonians/random-linear-regression/data
train_path = r'C:\Users\User\Downloads\archive\train.csv'
test_path = r'C:\Users\User\Downloads\archive\test.csv'

#Read the dataset
train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

#Clean dataset for null values(if any)
train = train.dropna()

#Set Hyperparameter values
iterations = 20
learning_rate = 0.0001

#Store data and labels separately
train_data = np.array(train.x).reshape(train.shape[0], 1)
train_labels = np.array(train.y).reshape(train.shape[0], 1)

test_data = np.array(test.x).reshape(test.shape[0], 1)
test_labels = np.array(test.y).reshape(test.shape[0], 1)

print("TrainData Dimensions: X: {} Y: {} \nTestData Dimensions: X: {} Y: {} \n"
      .format(train_data.shape, train_labels.shape, test_data.shape, test_labels.shape))

#Random initializations
weight = np.random.uniform(0, 1) * -1
bias = np.random.uniform(0, 1) * -1
plt.figure()

#Training
for iteration in range(iterations):
    
    #Forward propagation
    predictions = np.multiply(weight, train_data) + bias
    
    #CostFunction(MSE: Mean Squared Error)
    cost = np.mean((train_labels - predictions) ** 2) * 0.5
    print("Iteration: {}, Loss: {}".format(iteration+1, cost))
    
    #Plot the current status
    plt.plot(train_data, train_labels, '.')
    plt.plot(train_data, predictions, linewidth=2)
    plt.show()
    
    #Gradient Descent for back propagation
    cost_derivative = (train_labels - predictions) * -1
    derivative_wrt_weight = np.mean(np.multiply(train_data, cost_derivative))
    derivative_wrt_bias = np.mean(cost_derivative)
    
    #Update weight & bias parameters
    weight = weight - learning_rate * derivative_wrt_weight
    bias = bias - learning_rate * derivative_wrt_bias

#Testing
test_predictions = test_data * weight + bias
cost_test = np.mean((test_labels - test_predictions) ** 2) * 0.5
print("Model performance on test data \nCost: {}".format(cost_test))

plt.figure()
plt.plot(test_data, test_labels, '.')
plt.plot(test_data, test_predictions, linewidth=2)
plt.show()
