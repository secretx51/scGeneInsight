import torch
from torch import nn

class DeepGeneExpressionClassifier(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, dropout_prob=0.5, l1_reg=0.001):
        super(DeepGeneExpressionClassifier, self).__init__()

        self.input_layer = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0]),
            nn.BatchNorm1d(hidden_sizes[0]),
            nn.ReLU(),
            nn.Dropout(dropout_prob)
        )

        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_sizes) - 1):
            layer = nn.Sequential(
                nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]),
                nn.BatchNorm1d(hidden_sizes[i + 1]),
                nn.ReLU(),
                nn.Dropout(dropout_prob)
            )
            self.hidden_layers.append(layer)

        self.output_layer = nn.Linear(hidden_sizes[-1], output_size)
        self.l1_reg = l1_reg  # L1 regularization strength

    def forward(self, x):
        x = self.input_layer(x)
        for layer in self.hidden_layers:
            x = layer(x)
        x = self.output_layer(x)
        return x
    
    def l1_regularization(self):
        l1_loss = 0.0
        for param in self.parameters():
            l1_loss += torch.norm(param, p=1)
        return self.l1_reg * l1_loss
