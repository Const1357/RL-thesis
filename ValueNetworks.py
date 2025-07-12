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

        # if observation.dim() == 1:
        #     observation = observation.unsqueeze(0)  # ensure batched shape [B, ...]

        return self.network(observation)
    
class ValueCNN(ValueNetwork):
    def __init__(self, input_channels: int, input_height: int, input_width: int, conv_layers: list, fc_layers: list):
        super(ValueCNN, self).__init__()

        in_channels = input_channels

        # Convolutional Layers
        _conv_layers = []
        for out_channels, kernel_size, stride, padding in conv_layers:
            _conv_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding))
            _conv_layers.append(nn.ReLU())
            in_channels=out_channels
        self.conv = nn.Sequential(*_conv_layers)

        # flattened dimension computation by passing dummy input from the convolutional network
        with torch.no_grad():
            dummy_input = torch.zeros(1, input_channels, input_height, input_width)
            conv_out = self.conv(dummy_input)
            flat_dim = conv_out.view(1, -1).shape[1]

        # Fully Connected (Linear) Layers
        in_features = flat_dim
        _fc_layers = []
        for out_features in fc_layers:
            _fc_layers.append(nn.Linear(in_features, out_features))
            _fc_layers.append(nn.ReLU())
            in_features = out_features

        self.fc = nn.Sequential(*_fc_layers)

        self.head = nn.Linear(in_features, 1)     # output_size = 1 (predicted value)

    def forward(self, observation: torch.Tensor):

        if observation.dim() == 4:
            observation = observation.unsqueeze(0)  # ensure batched shape [B, ...]
        
        B, E, C, H, W = observation.shape

        x = observation.view(B*E, C, H, W)  # join batch and env dimensions, conv2d expects [B, C, H, W] input

        x = self.conv(x)
        x = x.view(x.size(0), -1)   # flattening
        x = self.fc(x)
        value = self.head(x).view(B, E, -1)    # converting [B x E, rest] -> [B, E, rest] to match the rest of the implementation
        # print(f"[Value]: {value[0,0].item()}")
        return value