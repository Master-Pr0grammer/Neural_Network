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

    # Forward Pass Definition
    def forward(input):
        return (1+np.exp(-1*input))**-1

    # Backward Pass Definition
    def backward(input):
        return np.exp(-input)/np.square((1+np.exp(-input)))

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

    # Forward Pass Definition
    def forward(input):
        return np.maximum(0, input)
    
    # Backward Pass Definition
    def backward(input):
        return np.heaviside(input, 0)

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

    Functions:
        forward() = takes in layer input and computes the layer forward probagation output
    '''

    def __init__(self, num_inputs, num_neurons):
        self.num_inputs=num_inputs
        self.num_neurons=num_neurons
        self.weights = 0.1 * np.random.rand(num_inputs, num_neurons)
        self.biases = np.zeros(num_neurons)

    # Forward Propagation Pass
    def forward(self, input, act_func):
        self.output = np.dot(input, self.weights) + self.biases
        self.activation = act_func.forward(self.output)
        return self.activation

# Neural Network Class Definition
class Neural_Network:
    '''
    Neural Network class definition used to create fully connected neural netorks very easily.

    Inputs:
        structure = a list where each position represents a separate layer, and each value (int) represents the number of neurons in that layer
        
        act_func = the activation function to be used for the network layers
        
        seed = (optional) the seed used for generating initial weights and biases, used to create predictable resulsts

    Functions:
        forward = runs forward pass of network on a given input and returns the output

        backpropagate = runs backpropagation pass on network

        evaluate = evaluates the network loss and accuracy on a given input and output dataset, and accuracy funciton
        
        save = saves current network state as a json file

        train = runs training loop on the network
    '''
    
    def __init__(self, structure=[], act_func=sigmoid, cost_func=MS_cost, seed=None):
        #set seed for randomly generated numbers
        if seed!=None:
            np.random.seed(seed)
        
        # build network
        self.structure = structure
        self.network = []
        self.act_func = act_func
        self.cost_func = cost_func

        for i in range(1, len(structure)):
            self.network.append(Layer(structure[i-1], structure[i]))

    # forward propagate
    def forward(self, input):
        for layer in self.network:
            output = layer.forward(input, self.act_func)
            input = output
        return output

    # back propagation
    def back_propagate(self, input, output, target, learning_rate):

        dcda = self.cost_func.backward(output, target)
        for i in range(len(self.network)-1, -1, -1):

            #find dz/dw and da/dz
            if i == 0:
                activation = input
                activation=np.reshape(activation, (len(activation),1))
                dzdw=activation
            else:
                activation=self.network[i-1].activation
                activation=np.reshape(activation, (len(activation),1))
                dzdw=activation

            dadz = self.act_func.backward(self.network[i].output)


            #calculate local layer weights/biases gradients
            temp=dcda * dadz
            temp=temp.reshape((1,len(temp)))
            weight_gradient = np.dot(dzdw, temp)

            #weight_gradient = dzdw * dadz * dcda
            bias_gradient = dadz * dcda


            #update local cost partial derivative to next layer (L-1)
            dzda=np.sum(self.network[i].weights, axis=0)
            dzda=(np.reshape(dzda, (1,len(dzda))) + self.network[i].biases).T
            dcda = np.dot((dadz * dcda), self.network[i].weights.T)

            self.network[i].weights -= learning_rate*weight_gradient
            self.network[i].biases -= learning_rate*bias_gradient

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
    
    def save(self, filename, loss = None, accuracy = None):
        network = {}
        network["Structure"] = self.structure
        network["Activation_Func"] = self.act_func.__name__
        network["Cost_Func"] = self.cost_func.__name__
        
        #get current weights/biases
        network_data=[]
        for i in range(len(self.network)):
            layer_weights = self.network[i].weights.tolist()
            layer_biases = self.network[i].biases.tolist()

            network_data.append((layer_weights, layer_biases))
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
                self.back_propagate(X[j], output, y[j], learning_rate)

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

#-------------------- EXAMPLE NEURAl NETWORK CLASS USAGE --------------------

# Example use of Neural Network Module
if __name__ == '__main__':

    #function for determining the accuracy of network
    def get_accuracy(output, target):
        output = output.reshape(len(target)).round()
        accuracy = np.mean(output == target)

        return accuracy

    #Training data example - XOR logic
    X = [[0, 0], [0, 1], [1, 0], [1, 1]]
    y = [0, 1, 1, 0]

    #Neural Network definition and training
    network = Neural_Network([2, 3, 1], act_func=sigmoid, cost_func=MS_cost, seed=0)
    network.train(X, y, 0.25, 10000, show_training_data=False, accuracy_func=get_accuracy)
    
    #Test Trained Model
    print('Training done, enter test values:')
    while(True):
        x1=int(input())
        x2=int(input())
        output = network.forward([x1,x2])
        if output[0]>0.5:
            print([x1,x2],'=', output, '=',1)
        else:
            print([x1,x2],'=',output,'=',0)