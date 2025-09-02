import torch.nn as nn

class MLP(nn.Module):
    """
    MLP với nhiều lớp hơn và Batch Normalization.
    """
    def __init__(self, input_dim, num_classes, hidden_dims=[1024, 512, 256]):
        super(MLP, self).__init__()
        
        layers = []
        current_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.5))
            current_dim = h_dim
            
        layers.append(nn.Linear(current_dim, num_classes))
        
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)