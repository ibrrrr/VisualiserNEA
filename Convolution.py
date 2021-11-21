import numpy as np
from layer import layer
from scipy import signal

class Convolutional(Layer):
    def __init__(self, input_shape, kernel_size, depth): #takes in tuple of depth height and width of input
        #it takes in kernel_size which represents the size of each matrix in each kernel
        #the depth represents how many kernels we want, therefore giving us the depth of the input
        input_depth, input_height, input_width = input_shape
        #unpack the input shape
        self.depth = depth
        self.input_shape = input_shape
        self.input_depth = input_depth
        self.output_shape = (depth, input_height - kernel_size + 1, input_width - kernel_size + 1)
        #computing the output shape, 3 dimensions, depth, height and width of output matrix computed using
        #size of the input - size of the kernel + 1
        self.kernels_shape = (depth, input_depth, kernel_size, kernel_size)
        #computes the shape of the kernels, 4 dimensions as we have multiple kernels each in 3 dimensions
        #1st is number of kernels (depth), depth of each kernel, size of the matrices contained in each kernel represented by last two
        self.kernels = np.random.randn(*self.kernels_shape)
        self.biases = np.random.randn(*self.output_shape)
        #initialise the kernels and biases randomly using a randn function.

def forward(self, input):
    #takes in the input and computes a formula (shown in documentation)
    self.input = input
    self.output = np.copy(self.biases)
    #copy the current biases as each bias is equal to the bias + something else
    for i in range(self, depth)
        #loop through output depth
        for j in range(self.input_depth):
            #loop through input depth and continuously add values
            self.output[i] += signal.correlate2d(self.input[j], self.kernels[i, j], "valid")
            #this function is not commutative meaning the order of our inputs here matter
            return self.output

def backward(self, output_gradient, learning_rate):
    kernels_gradient = np.zeros(self.kernels_shape)
    input_gradient = np.zeros(self.input_shape)

    for i in range(self.depth):
        for j in range(self.input_depth):
            kernels_gradient[i, j] = signal.correlate2d(self.input[j], output_gradient[i], "valid")
            input_gradient[j] += signal.convolve2d(output_gradient[i], self.kernels[i, j], "full")

        self.kernels -= learning_rate * kernels_gradient
        self.biases -= learning_rate * output_gradient
        return input_gradient

    