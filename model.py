import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc = nn.Linear(10, 2)  # Example: 10 input features → 2 classes

    def forward(self, x):
        return self.fc(x)
####not necessary