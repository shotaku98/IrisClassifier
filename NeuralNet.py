#####################################################################################################################
#   CS 6375 - Assignment 2, Neural Network Programming
#   This is a starter code in Python 3.6 for a 2-hidden-layer neural network.
#   You need to have numpy and pandas installed before running this code.
#   Below are the meaning of symbols:
#   train - training dataset - can be a link to a URL or a local file
#         - you can assume the last column will the label column
#   train - test dataset - can be a link to a URL or a local file
#         - you can assume the last column will the label column
#   h1 - number of neurons in the first hidden layer
#   h2 - number of neurons in the second hidden layer
#   X - vector of features for each instance
#   y - output for each instance
#   w01, delta01, X01 - weights, updates and outputs for connection from layer 0 (input) to layer 1 (first hidden)
#   w12, delata12, X12 - weights, updates and outputs for connection from layer 1 (first hidden) to layer 2 (second hidden)
#   w23, delta23, X23 - weights, updates and outputs for connection from layer 2 (second hidden) to layer 3 (output layer)
#
#####################################################################################################################


import pandas as pd
import numpy as np


class NeuralNet:
    def __init__(self, train, activation="sigmoid", header=True, h1=4, h2=2):
        np.random.seed(1)
        # train refers to the training dataset
        # test refers to the testing dataset
        # h1 and h2 represent the number of nodes in 1st and 2nd hidden layers

        raw_input = pd.read_csv(train)
        self.activation = activation

        train_dataset = self.preprocess(raw_input)
        ncols = len(train_dataset.columns)
        nrows = len(train_dataset.index)
        self.X = train_dataset.iloc[:, 0:(ncols - 3)].values.reshape(nrows, ncols - 3)
        self.y = train_dataset.iloc[:, (ncols - 3):].values.reshape(nrows, 3)
        #
        # Find number of input and output layers from the dataset
        #
        input_layer_size = len(self.X[0])
        if not isinstance(self.y[0], np.ndarray):
            output_layer_size = 1
        else:
            output_layer_size = len(self.y[0])

        # assign random weights to matrices in network
        # number of weights connecting layerss = (no. of nodes in previous layer) x (no. of nodes in following layer)
        self.w01 = 2 * np.random.random((input_layer_size, h1)) - 1
        self.X01 = self.X
        self.delta01 = np.zeros((input_layer_size, h1))
        self.w12 = 2 * np.random.random((h1, h2)) - 1
        self.X12 = np.zeros((len(self.X), h1))
        self.delta12 = np.zeros((h1, h2))
        self.w23 = 2 * np.random.random((h2, output_layer_size)) - 1
        self.X23 = np.zeros((len(self.X), h2))
        self.delta23 = np.zeros((h2, output_layer_size))
        self.deltaOut = np.zeros((output_layer_size, 1))

    #
    # The activation function for sigmoid, tanh and relu
    #

    def __activation(self, x):
        if self.activation == "sigmoid":
            return self.__sigmoid(x)
        elif self.activation == "tanh":
            return self.__tanh(x)
        elif self.activation == "reLu":
            return self.__reLu(x)

    #
    # The activation derivative function for sigmoid, tanh and relu
    #

    def __activation_derivative(self, x):
        if self.activation == "sigmoid":
            return self.__sigmoid_derivative(x)
        elif self.activation == "tanh":
            return self.__tanh_derivative(x)
        elif self.activation == "reLu":
            return self.__reLu_derivative(x)

    def __sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # derivative of sigmoid function, indicates confidence about existing weight

    def __sigmoid_derivative(self, x):
        return x * (1 - x)

    def __tanh(self, x):
        ex = np.exp(x)
        e_minus_x = np.exp(-x)
        return (ex - e_minus_x) / (ex + e_minus_x)

    def __tanh_derivative(self, x):
        return 1 - np.power(x, 2)

    def __reLu(self, x):
        zeros = np.zeros(x.shape)
        out = np.maximum(zeros, x)
        return out

    def __reLu_derivative(self, x):
        x[x > 0] = 1
        x[x < 0] = 0
        return x

    #
    #  Pre-processing the dataset, which would include standardization, normalization,
    #   categorical to numerical, etc
    #

    def preprocess(self, X):
        X = X.dropna()
        X = X.drop_duplicates()
        # The output layer is having 3 neurons as there are 3 different classes.
        X["Y1"] = (X["Y"] == "Iris-setosa") * 1
        X["Y2"] = (X["Y"] == "Iris-versicolor") * 1
        X["Y3"] = (X["Y"] == "Iris-virginica") * 1
        X = X.drop("Y", 1)
        # X.loc[X["Y"] == "Iris-setosa", "Y"] = np.ndarray([1, 0, 0])
        # X.loc[X["Y"] == "Iris-versicolor", "Y"] = np.ndarray([0, 1, 0])
        # X.loc[X["Y"] == "Iris-virginica", "Y"] = np.ndarray([0, 0, 1])
        # Scaling the features between 0 and 1.
        for column in X.columns[0:4]:
            X[column] = X[column] - X[column].min()
            X[column] = X[column] / X[column].max()

        return X

    # Training function

    def train(self, max_iterations=1000, learning_rate=0.05):
        error = 0
        for iteration in range(max_iterations):
            out = self.forward_pass()
            error = 0.5 * np.power((out - self.y), 2)
            self.backward_pass(out)
            update_layer2 = learning_rate * self.X23.T.dot(self.deltaOut)
            update_layer1 = learning_rate * self.X12.T.dot(self.delta23)
            update_input = learning_rate * self.X01.T.dot(self.delta12)

            self.w23 += update_layer2
            self.w12 += update_layer1
            self.w01 += update_input

        print("After " + str(max_iterations) + " iterations, the total error is " + str(np.sum(error)))
        print("The final weight vectors are (starting from input to output layers)")
        print(self.w01)
        print(self.w12)
        print(self.w23)

    def forward_pass(self):
        # pass our inputs through our neural network
        in1 = np.dot(self.X, self.w01)
        self.X12 = self.__activation(in1)
        in2 = np.dot(self.X12, self.w12)
        self.X23 = self.__activation(in2)
        in3 = np.dot(self.X23, self.w23)
        out = self.__activation(in3)
        return out

    def backward_pass(self, out):
        # pass our inputs through our neural network
        self.compute_output_delta(out)
        self.compute_hidden_layer2_delta()
        self.compute_hidden_layer1_delta()

    # Computing the deltas

    def compute_output_delta(self, out):
        delta_output = (self.y - out) * (self.__activation_derivative(out))
        self.deltaOut = delta_output

    def compute_hidden_layer2_delta(self):
        delta_hidden_layer2 = (self.deltaOut.dot(self.w23.T)) * (self.__activation_derivative(self.X23))
        self.delta23 = delta_hidden_layer2

    def compute_hidden_layer1_delta(self):
        delta_hidden_layer1 = (self.delta23.dot(self.w12.T)) * (self.__activation_derivative(self.X12))
        self.delta12 = delta_hidden_layer1

    def compute_input_layer_delta(self):
        delta_input_layer = np.multiply(self.__activation_derivative(self.X01), self.delta01.dot(self.w01.T))
        self.delta01 = delta_input_layer

    # Implementation of predict function
    # Assuming that the test dataset has the same format as the training dataset
    # output the test error from this function

    def predict(self, test, header=True):
        testDF = pd.read_csv(test)
        testDF = self.preprocess(testDF)
        ncols = len(testDF.columns)
        nrows = len(testDF.index)
        self.X = testDF.iloc[:, 0:(ncols - 3)].values.reshape(nrows, ncols - 3)
        self.y = testDF.iloc[:, (ncols - 3):].values.reshape(nrows, 3)
        out = self.forward_pass()
        error = np.power((out - self.y), 2)
        test_error = 0.5 * np.sum(error)
        print("accuracy = " + str(100 - test_error))
        return test_error


if __name__ == "__main__":
    activations = ["sigmoid", "tanh", "reLu"]
    for activation in activations:
        neural_network = NeuralNet("iris.train.csv", activation)
        neural_network.train(max_iterations=1000, learning_rate=0.01)
        testError = neural_network.predict("iris.test.csv")
        print("Test Error is equal : " + str(testError))
