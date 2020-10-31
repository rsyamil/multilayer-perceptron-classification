import csv
import numpy as np
import scipy.special as sp

#load all data needed
def loadData(dir, verbose=False):
    data = np.genfromtxt(dir, delimiter=',')
    if verbose:
        print("Read " + dir + " dimension " + str(data.shape))
    return data
    
#write to output file
def writeOutput(dir, data, verbose=False):
    np.savetxt(dir, data, fmt = "%d")
    if verbose:
        print("Writing output to .csv file")
    
#neural network - multi layer perceptron
class NeuralNetwork:
    
    def __init__(self, input_nodes_sz=None, hidden_nodes_sz=None, 
                output_nodes_sz=None, learning_rate=None, epochs=None, activation=None, verbose=None):
        self.verbose = verbose
        #architecture parameter
        self.input_sz = input_nodes_sz
        self.hidden_sz = hidden_nodes_sz
        self.output_sz = output_nodes_sz
        #training parameter
        self.learning_rate = learning_rate
        self.epochs = epochs
        #mlp weights to be initialized
        self.w1 = []
        self.w2 = []
        self.activation = activation
        
    def initialize(self, initializer=None):
        if initializer == "random":
            #xavier-he-like initialization
            self.w1 = np.random.normal(0.0, 1.0/np.sqrt(self.hidden_sz), [self.input_sz, self.hidden_sz])   #784x300
            self.w2 = np.random.normal(0.0, 1.0/np.sqrt(self.output_sz), [self.hidden_sz, self.output_sz])  #300x10
        else:
            print("To be implemented")
            
    def activationFunction(self, input):
        if self.activation == "sigmoid":
            return sp.expit(input)
        else:
            print("To be implemented")
            
    def oneHot(self, labels):
        labels_onehot = np.zeros([labels.shape[0], np.max(labels) + 1])
        labels_onehot[np.arange(labels.shape[0]), labels] = 1
        return labels_onehot
        
    def normalizeInput(self, input):
        input_normalized = input/255.0
        return input_normalized
    
    def train(self, train_image, train_label):
        train_label_onehot = self.oneHot(train_label)
        train_image_normalized = self.normalizeInput(train_image)
        for i in range(self.epochs):
            for image, label in zip(train_image_normalized, train_label_onehot):
                image = np.expand_dims(image, axis=-1).T        #1x784
                label = np.expand_dims(label, axis=-1).T        #1x10
                
                h_ = self.activationFunction(np.dot(image, self.w1))    #1x300
                o_ = self.activationFunction(np.dot(h_, self.w2))       #1x10
                
                o_error = label - o_                                    #1x10
                h_error = np.dot(o_error, self.w2.T)                    #1x300
                
                #calculate update to the weight matrix
                delta_w1 = np.dot(image.T, (h_error * h_ * (1 - h_)))   #783x300
                delta_w2 = np.dot(h_.T, (o_error * o_ * (1 - o_)))      #300x10
                
                #update the weights
                self.w1 = self.w1 + (self.learning_rate * delta_w1)
                self.w2 = self.w2 + (self.learning_rate * delta_w2)
        if self.verbose:
            print("Training completed")
                
    def test(self, test_image):
        test_image_normalized = self.normalizeInput(test_image)        
        h_ = self.activationFunction(np.dot(test_image_normalized, self.w1))    #1x300
        o_ = self.activationFunction(np.dot(h_, self.w2))       #1x10
        if self.verbose:
            print("Testing performance : " + str(o_.shape))
        return o_ 
                            
if __name__ == "__main__":

    #flags
    verbose = False
    debug = False

    #read the data from .csv files
    dirs = ["train_image.csv", "test_image.csv", "train_label.csv"]
    train_image = loadData(dirs[0], verbose)
    test_image = loadData(dirs[1], verbose)
    train_label = loadData(dirs[2], verbose)
    
    #define neural network dimension parameter
    input_nodes_sz = train_image.shape[-1]
    hidden_nodes_sz = 300
    output_nodes_sz = 10
    
    #define training parameter
    initializer = "random"
    activation = "sigmoid"
    learning_rate = 0.01 * 5
    epochs = 5
    
    #create instance of neural network and initialize
    mlp = NeuralNetwork(input_nodes_sz, hidden_nodes_sz, output_nodes_sz, learning_rate, epochs, activation, verbose)
    mlp.initialize(initializer=initializer)    
    
    #train the network
    mlp.train(train_image, train_label.astype('int32'))
    
    #get test output and write to file
    test_label_hat = (np.argmax(mlp.test(test_image), axis=-1)).astype('int32')
    writeOutput("test_predictions.csv", test_label_hat, verbose)
    
