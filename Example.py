from Neural_Network import *

#function for determining the accuracy of network
def get_accuracy(output, target):
    output = output.reshape(len(target)).round()
    accuracy = np.mean(output == target)

    return accuracy

#Training data example - XOR logic
X = [[0, 0], [0, 1], [1, 0], [1, 1]]
y = [0, 1, 1, 0]

#Neural Network definition and training
network = Neural_Network([2, 3, 1], act_func=sigmoid, cost_func=MS_cost)
network.train(X, y, 0.25, 4000, show_training_data = True, seed = 0, accuracy_func = get_accuracy)

#Test Trained Model
while(True):
    x1, x2=input("Enter two input values 1 or 0, separated by a space: ").split()
    output = network.forward([int(x1),int(x2)])

    if output[0]>0.5:
        print([x1,x2],'=', output, '=',1)
    else:
        print([x1,x2],'=',output,'=',0)