import numpy as np

class ArtificialNeuralNetwork():
    def __init__(self, stateDim = 4, actionDim = 1, hiddenNodes = 8, lr = 0.05):
        # Store dimensions for state and action
        self.stateDim = 4
        self.actionDim = 1

        # Variables for learning
        self.lr = lr # Learning Rate

        # Create Number of layers for hidden nodes
        self.layer1Nodes = hiddenNodes
        self.layer2Nodes = 2*hiddenNodes

        #Create Architecture
        self.layers = self.createArchitecture()

    def createArchitecture(self):
        '''
        Creates and returns list of layers in network with random initial values
        '''
        # Initialize weights to random sizes
        layer1 = np.random.rand(self.stateDim+1, self.layer1Nodes) * np.sqrt(2/self.stateDim+1)
        layer2 = np.random.rand(self.layer1Nodes+1, self.layer2Nodes) * np.sqrt(2/self.layer1Nodes) 
        layer3 = np.random.rand(self.layer2Nodes+1, self.actionDim) * np.sqrt(2/self.layer2Nodes)

        # Return list of layers

        return (layer1, layer2, layer3)
    
    def ReLU(self,arr):
        '''
        Simple implementation of Rectified Linear Unit activation function.
        Returns 0 if value is negative or value if it is positive
        '''
        # Change all values less than equal to 0 to zero
        arr[arr<0] = 0
        return arr

    def update(self, state, y):
        '''
        Update the values of the weights given state and outcome y
        '''
        # Get predicted q-value from neural network model
        steps, output = self.predict(state)
        
        # Compute required update for weights
        delta = self.backPropagation(steps, output, y)

    def backPropagation(self, steps, output, y):
        '''
        Compute backpropagation in neural network to determine required deltas for each weight
        '''
        print('Back Propagation Start')


    def predict(self,state):
        '''
        Given a state, predict the Q-Values of each action
        '''
        # Create array to store intermediate steps
        steps = []

        # Get predicted q-value from neural network model
        output = state
        steps.append(output)

        for i in range(len(self.layers)): # Iterate through all layers by index
            # Add bias term to output
            output = np.array([1, *output])

            # Multiply by layer weights
            output = np.matmul(output.transpose(), self.layers[i])
            
            # Store intermediate steps in array
            steps.append(output)

            # For all layers except last, use activation function self.ReLU()
            if i != len(self.layers):
                output = self.ReLU(output)

        # Return output
        return steps, output

def main():
    ann = ArtificialNeuralNetwork()
    ann.update(np.array([0.5, 0.5, 0.5, 0.5]), 1)


if __name__ == '__main__':
    main()
