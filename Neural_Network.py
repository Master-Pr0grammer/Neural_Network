import copy, json
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

#-------------------- ACTIVATION FUNCTIONS --------------------

# sigmoid activation function
class sigmoid:
    '''
    Uses sigmoid non linearity function to transform
    Neural network layer outputs

    Functions:
        Forward() = determines regular output of sigmoid for forward propagation
        
        Backward() = determines input derivative output of sigmoid for back propagation

    Arguments:
        Input = numpy array of inputs to sigmoid function

    Output:
        numpy array containing sigmoid function ouputs of the inputs in the input array
    '''

    def __init__(self):
        self.name = 'sigmoid'

    # Forward Pass Definition
    def forward(self, input):
        self.output = (1+np.exp(-1*input))**-1
        return self.output

    # Backward Pass Definition
    def backward(self):
        return self.output * (1 - self.output)

# Rectified Linear Activation Function
class ReLU:
    '''
    Uses rectified linear (ReLU) non linearity function to transform
    Neural network layer outputs

    Functions:
        Forward() = determines regular output of ReLU for forward propagation
        
        Backward() = determines input derivative output of ReLU for back propagation

    Arguments:
        Input = numpy array of inputs to ReLU function (ussually network layer output)

    Output:
        numpy array containing ReLU function ouputs of the inputs in the input array
    '''

    def __init__(self):
        self.name = 'ReLU'

    # Forward Pass Definition
    def forward(self, input):
        self.output = np.maximum(0, input)
        return np.maximum(0, input)
    
    # Backward Pass Definition
    def backward(self):
        return np.heaviside(self.output, 0)

#Softmax Function
class softmax:
    '''
    Uses Softmax function to transform Neural network layer outputs into a set of
    relative probabilities for clasification networks

    Functions:
        Forward() = determines regular output of softmax for forward propagation
        
        Backward() = determines input derivative output of softmax for back propagation

    Arguments:
        Input = numpy array of inputs to softmax function (ussually final network layer output)

    Output:
        numpy array containing softmax function ouputs of the inputs in the input array
    '''

    def __init__(self):
        self.name = 'softmax'

    # Forward Pass Definition
    def forward(input):
        e_val = np.exp(input - np.max(input, axis=1, keepdims=True))
        probability = e_val/np.sum(e_val, axis=1, keepdims=True)
        return probability
    
    # Backward Pass Definition
    def backwards(input):
        s = softmax(input)
        d = np.outer(s, (1 - s))
        np.fill_diagonal(d, s * (1 - s))
        return d

#-------------------- Cost FUNCTIONS --------------------

# mean square cost function
class MS_cost:
    '''
    Uses mean squared cost function to get a loss for a neural netowork ouptut

    Functions:
        Forward() = determines regular output of mean squared cost function for forward propagation
        
        Backward() = determines input derivative output of mean squared cost function for back propagation

    Arguments:
        Input = numpy array of inputs to mean squared cost function function (ussually final network layer output)

    Output:
        numpy array containing mean squared cost function function ouputs of the inputs in the input array
    '''

    # Forward Pass Definition
    def forward(output, target):
        output1 = copy.deepcopy(output)

        if len(output1) == 1:
            return np.square(output1-target)
        else:
            for i in range(len(target)):
                output1[i] -= target[i]
            return np.square(output1)

    # Backward Pass Definition
    def backward(output, target):
        output1 = copy.deepcopy(output)
        if len(output1)== 1:
            output1-= target
            return 2*output1
        else:
            length = len(output1)
            for i in range(length):
                output1[i] -= target[i]
            return 2*output1

#-------------------- NEURAl NETWORK DEFINITION --------------------

# Neural Network Layer Definition
class Layer:
    '''
    Layer class definition used in Neural Network class. Creates a neural network layer
    that can be combined with other layers to create a network

    Input:
        num_inputs = Number of inputs into layer (ussually the number of outputs from the previous layer, or number of network inputs if this is the first layer)
        
        num_neurons = Number of neurons in layer

        act_func = determines the activation function that will be used on the layer

    Functions:
        forward() = takes in layer input and computes the layer forward probagation output
    '''

    def __init__(self, num_inputs, num_neurons, act_func = sigmoid):
        self.num_inputs = num_inputs
        self.num_neurons = num_neurons
        self.act_func = act_func()
        self.weights = 0.1 * np.random.rand(num_inputs, num_neurons)
        self.biases = np.zeros(num_neurons)

    # Forward Propagation Pass
    def forward(self, input):
        self.input = input
        self.output = np.dot(input, self.weights) + self.biases
        self.activation = self.act_func.forward(self.output)
        return self.activation
    
    # Backward Propagation Pass
    def backward(self, learning_rate, dcda):
        if self.input.ndim == 1:
            dzdw = self.input.reshape(self.input.shape[0], 1)
        else:
            dzdw = self.input.reshape(self.input.shape[0], self.input.shape[1], 1)

        dadz = self.act_func.backward()

        #calculate local layer weights/biases gradients
        dcdz = dcda * dadz
        if self.input.ndim == 1:
            dcdz = dcdz.reshape(1,dcdz.shape[0])
        else: 
            dcdz = dcdz.reshape(dcdz.shape[0], 1, dcdz.shape[1])
            dcdz = dcdz.mean(axis=0)

        weight_gradient = np.dot(dzdw, dcdz)
        bias_gradient = dadz * dcda
        
        if self.input.ndim != 1:
            weight_gradient = weight_gradient.mean(axis=0)
            bias_gradient = bias_gradient.mean(axis=0)


        #update layer weights/biases
        self.weights -= learning_rate * weight_gradient
        self.biases -= learning_rate * bias_gradient


        #update local cost partial derivative for previous layer (L-1)
        dcda = np.dot((dadz * dcda), self.weights.T)
        return dcda

# Neural Network Class Definition
class Neural_Network:
    '''
    Neural Network class definition used to create fully connected neural netorks very easily. Inputs Must be numpy arrays

    Inputs:
        structure = a list where each position represents a separate layer, and each value (int) represents the number of neurons in that layer, with the exception of the first layer being a "dummy" input neuron layer
        
        act_func = the activation function to be used for the network layers. This can be changed after declaring the network on a per layer basis so different layers can have different funcitons
        
        seed = (optional) the seed used for generating initial weights and biases, used to create predictable resulsts

    Functions:
        forward = runs forward pass of network on a given input and returns the output, can handle batch inputs aswell.

        backward = runs backpropagation pass on network, can also handle batch inputs

        evaluate = evaluates the network loss and accuracy on a given input and output dataset, and accuracy funciton
        
        save = saves current network state as a json file

        train = runs training loop on the network, trains on individual samples. However, the network does support batches
    '''
    
    def __init__(self, structure = [], act_func = sigmoid, cost_func = MS_cost, seed = None):
        #set seed for randomly generated numbers
        if seed!=None:
            np.random.seed(seed)
        
        # build network
        self.structure = structure
        self.network = []
        self.cost_func = cost_func

        for i in range(1, len(structure)):
            self.network.append(Layer(structure[i-1], structure[i], act_func = act_func))

    # forward propagation
    def forward(self, input):
        for layer in self.network:
            output = layer.forward(input)
            input = output
        return output

    # backward propagation
    def backward(self, output, target, learning_rate):
        dcda = self.cost_func.backward(output, target)
        for i in range(len(self.network)-1, -1, -1):
            dcda = self.network[i].backward(learning_rate, dcda)

    # evaluates average cost and accuracy over whole data set
    def evaluate(self, X, y, accuracy_func=None):
        predictions = []
        costs = []
        for i in range(len(X)):
            output = self.forward(X[i])
            predictions.append(output)
            costs.append(self.cost_func.forward(output, y[i]))
        predictions = np.array(predictions)

        if accuracy_func!=None:
            accuracy = accuracy_func(predictions, y)
        else:
            accuracy = None

        return accuracy, np.mean(costs)
    
    # save current neural network state as a json file
    def save(self, filename, loss = None, accuracy = None):
        network = {}
        network["Structure"] = self.structure
        network["Cost_Func"] = self.cost_func.__name__
        
        #get current weights/biases
        network_data=[]
        for i in range(len(self.network)):
            layer_weights = self.network[i].weights.tolist()
            layer_biases = self.network[i].biases.tolist()
            act_func_name = self.network[i].act_func.name

            network_data.append([layer_weights, layer_biases, act_func_name])
        network["Data"] = network_data

        #optionally save Loss/Accuracy data
        if loss != None:
            network["Loss"] = loss

        if accuracy != None:
            network["Accuracy"] = accuracy

        #write to file to save netowork
        file = open(filename, 'w')
        json.dump(network, file)
        file.close()

    # run training
    def train(self, X, y, learning_rate, num_epoch, show_training_data=False, accuracy_func=None):
        pbar=tqdm(desc='Training Neural Network', total=num_epoch)
        loss_data = []
        accuracy_data = []

        #main training loop
        for i in range(num_epoch):
            average_cost=[]
            for j in range(len(X)):
                output = self.forward(X[j])
                average_cost.append(self.cost_func.forward(output, y[j]))
                self.backward(output, y[j], learning_rate)

            #evaluate model
            accuracy, loss = self.evaluate(X, y, accuracy_func=accuracy_func)
            loss_data.append(loss)

            if accuracy_func!=None:
                accuracy_data.append(accuracy)
                pbar.set_description(f"Loss: {loss:.3f}, Accuracy: {accuracy:.3f}")
            else:
                pbar.set_description(f"Loss: {loss:.3f}")

            pbar.update()
        pbar.close()

        # plot data
        if show_training_data:
            fig, ax = plt.subplots(2)
            ax[0].plot(loss_data)
            ax[0].set_title('Cost Vs Epoch')
            ax[1].plot(accuracy_data)
            ax[1].set_title('Accuracy Vs Epoch')
            fig.tight_layout()
            plt.show()
