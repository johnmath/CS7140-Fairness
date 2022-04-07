import torch.nn as nn

class NeuralNet(nn.Module):
    
    def __init__(self, input_dim, layer_sizes=[2,4,8], num_classes=2):
        super(NeuralNet, self).__init__()
        self._input_dim = input_dim
        self._layer_sizes = layer_sizes
        self._num_classes = num_classes
    
        layers = [nn.Linear(input_dim, layer_sizes[0])]
        layers.append(nn.ReLU())
        layers.append(nn.Dropout())
        
        # Initialize all layers according to sizes in list
        for i in range(len(self._layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout())
        layers.append(nn.Linear(layer_sizes[-1], num_classes))
        
        # Wrap layers in ModuleList so PyTorch
        # can compute gradients
        self.layers = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.layers(x)
    


