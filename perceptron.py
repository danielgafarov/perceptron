import numpy as np

class Perceptron():

    def __init__(self):
        self.synaptic_weights = np.random.random(3) - 1
        
    #sigmoid function
    def sigmoid(self, x):
        return 1/(1 + np.exp(-x))

    #derivative of sigmoid function
    def sigmoid_derivative(self, x):
        Sx = self.sigmoid(x)
        return Sx * (1 - Sx)
    
    #training loop
    def train(self, inputs, targets, iterations):
        for i in range(iterations):
            outputs = self.think(inputs)
            Eout = targets - outputs
            adjustments = np.dot(inputs.T, Eout * self.sigmoid_derivative(outputs))
            self.synaptic_weights += adjustments 
    
    #one calculation step of the perceptron
    def think(self, inputs):
        return self.sigmoid(np.dot(inputs,self.synaptic_weights))
        
        
if __name__ == "__main__":
    p =  Perceptron()
    print(f"Synaptic weights before training: {p.synaptic_weights}")
    inputs = np.array([[0, 0, 1],
                       [1, 1, 1],
                       [1, 0, 0],
                       [0, 1, 1]])
    p.train(inputs,np.array([0,1,1,0]),100000)
    print(f"Synaptic weights after training: {p.synaptic_weights}")
    I1 = input("Enter value for the signal I1: ")
    I2 = input("Enter value for the signal I2: ")
    I3 = input("Enter value for the signal I3: ")
    print(p.think([int(I1),int(I2),int(I3)]))