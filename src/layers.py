from src.neuron import Neuron

class Layer:
    def __init__(self, size):
        self.neurons = [ Neuron()  * size] 

