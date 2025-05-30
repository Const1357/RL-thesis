from Network import *

class ValueMLP(ValueNetwork):
    def __init__(self,
                 input_size,
                 hidden_layers,
                 name = ''):
        super(ValueMLP, self).__init__()

        self.input_size = input_size
        self._name = name

        # Network that outputs single scalar V_pred(s)
        layers = []
        last_size = self.input_size
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(last_size, hidden_size))
            layers.append(nn.LeakyReLU())
            last_size = hidden_size
        layers.append(nn.Linear(last_size, 1))
        self.network = nn.Sequential(*layers).to(device)

    def forward(self, observation: torch.Tensor):

        if observation.dim() == 1:
            observation = observation.unsqueeze(0)  # ensure batched shape [B, ...]

        return self.network(observation)
    
class ValueCNN(ValueNetwork):
    pass