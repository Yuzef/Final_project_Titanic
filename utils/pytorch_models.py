from torch import nn

def get_activation(name):
    if name == "relu":
        return nn.ReLU()

    if name == "tanh":
        return nn.Tanh()

    if name == "sigmoid":
        return nn.Sigmoid()

    else:
        raise ValueError(f"Unknown activation: {name}.")

# output_dim = 2 (бинарная классификация).
class TitanicMLPNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, activation_name):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            get_activation(activation_name),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.layers(x)


