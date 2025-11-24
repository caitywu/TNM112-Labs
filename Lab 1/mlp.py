import numpy as np
import data_generator

# Different activations functions
def activation(x, activation):
    
    # 'activation' could be: 'linear', 'relu', 'sigmoid', or 'softmax'
    if activation == 'linear':
        return x
    elif activation == 'relu':
        return np.maximum(0.0,x)
    elif activation == 'sigmoid':
        return 1/(1 + np.exp(-x))
    elif activation == 'softmax':
        return np.exp(x) / np.sum(np.exp(x), axis=0)

    else:
        raise Exception("Activation function is not valid", activation) 

#-------------------------------
# Our own implementation of an MLP
#-------------------------------
class MLP:
    def __init__(
        self,
        dataset,         # DataGenerator
    ):
        self.dataset = dataset

    # Set up the MLP from provided weights and biases
    def setup_model(
        self,
        W,                   # List of weight matrices
        b,                   # List of bias vectors
        activation='linear'  # Activation function of layers
    ):
        self.activation = activation
        self.hidden_layers = len(W)-1

        self.W = W
        self.b = b

        #specify the total number of weights in the model (both weight matrices and bias vectors)
        self.N = sum(Wi.size for Wi in W) + sum(bi.size for bi in b)

        print('Number of hidden layers: ', self.hidden_layers)
        print('Number of model weights: ', self.N)

    # Feed-forward through the MLP
    def feedforward(
        self,
        x      # Input data points
    ):
        # TODO: specify a matrix for storing output values
        y = np.zeros((len(x), self.dataset.K))

        # TODO: implement the feed-forward layer operations
        a = 0

        # 1. Specify a loop over all the datapoints
        for i, x_i in enumerate(x):

            # 2. Specify the input layer (2x1 matrix)
            a = np.reshape(x_i,(2,1))

            # 3. For each hidden layer, perform the MLP operations
            for layer in range(self.hidden_layers):

                #    - multiply weight matrix and output from previous layer
                #    - add bias vector
                a = np.dot(self.W[layer], a) + self.b[layer]

                #    - apply activation function
                a = activation(a, self.activation)

            # 4. Specify the final layer, with 'softmax' activation
            a = np.dot(self.W[-1], a) + self.b[-1]
            a = activation(a, 'softmax')
            y[i] = a.flatten()
            
        return y

    # Measure performance of model
    def evaluate(self):
        print('Model performance:')

        # Assume the mean squared error loss
        # Hint: For calculating accuracy, use np.argmax to get predicted class
        out_train = self.feedforward(self.dataset.x_train)
        out_test = self.feedforward(self.dataset.x_test)

        train_loss = np.mean((out_train - self.dataset.y_train_oh)**2)
        train_acc = np.mean(np.argmax(out_train, 1) == self.dataset.y_train) * 100

        print("\tTrain loss:     %0.4f"%train_loss)
        print("\tTrain accuracy: %0.2f"%train_acc)

        # TODO: formulate the test loss and accuracy of the MLP
        test_loss = np.mean((out_test - self.dataset.y_test_oh)**2)
        test_acc = np.mean(np.argmax(out_test, 1) == self.dataset.y_test) * 100
        print("\tTest loss:      %0.4f"%train_loss)
        print("\tTest accuracy:  %0.2f"%test_acc)
