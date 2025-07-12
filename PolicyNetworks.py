from Network import *


# Trivial Approach: NN generates logits for each action. Distribution from logits.
class LogitsMLP(PolicyNetwork):
    def __init__(self,
                 input_size,
                 output_size,
                 hidden_layers,
                 name = '',):
        super(LogitsMLP, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self._name = name

        # Network that outputs output_size values
        layers = []
        last_size = self.input_size
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(last_size, hidden_size))
            layers.append(nn.LeakyReLU())
            last_size = hidden_size
        layers.append(nn.Linear(last_size, output_size))
        self.network = nn.Sequential(*layers).to(device)

    def forward(self, observation: torch.Tensor)->torch.Tensor:

        # if observation.dim() == 1:
        #     observation = observation.unsqueeze(0)

        logits = self.network(observation)
        probs = Categorical(logits = logits)

        return probs, logits
    
# GNN
class GNN_MLP(PolicyNetwork):
    def __init__(self,
                 input_size,
                 hidden_layers,
                 intervals, # list of size env.action_space.n (N) containing pairs (x_from, x_to)
                 mapping,   # list of size env.action_space.n (N) containing unique integers from 0 to N-1
                 name = '',):
        super(GNN_MLP, self).__init__()

        self.input_size = input_size
        self.output_size = 2            # mean, variance
        self._name = name

        self.register_buffer("intervals", torch.tensor(intervals, dtype=torch.float32, device=device).unsqueeze_(0))   # [1,N,2] to be transformed to [B,N,2]
        self.register_buffer("mapping", torch.tensor(mapping, dtype=torch.long, device=device))          # (N)

        self.xmin = min(intervals, key=lambda x : x[0])[0]     # leftmost in the interval
        self.xmax = max(intervals, key=lambda x : x[1])[1]     # rightmost in the interval

        # Network that outputs 2 values: mean and std
        layers = []
        last_size = self.input_size
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(last_size, hidden_size))
            layers.append(nn.LeakyReLU())
            last_size = hidden_size
        layers.append(nn.Linear(last_size, self.output_size))
        self.network = nn.Sequential(*layers).to(device)

    def forward(self, observation: torch.Tensor):
        
        # B, E, O = observation.shape                     # [B, E, O]
        B, O = observation.shape                        # [B, O]

        out = self.network(observation)                 # [B, 2] -> 2= 0.μ, 1.σ

        # expanded to match [B, N, K]
        mean = out[:, 0].unsqueeze(-1).unsqueeze(-1)     # [B, 1, 1]

        std = out[:, 1].unsqueeze(-1).unsqueeze(-1)      # [B, 1, 1]
        std = fn.softplus(std) + tol                    # [B, 1, 1] softplus for numerical stability
        std.clamp_min_(tol)

        intervals = self.intervals.expand(B, -1, -1)    # [B, N, 2] 

        x_from = intervals[:, :,0].unsqueeze(-1)       # [B, N, 1]
        x_to   = intervals[:, :,1].unsqueeze(-1)       # [B, N, 1]

        # Integrating over intervals of interest
        probs = gaussian_integral(mean, std, x_from=x_from, x_to=x_to)  # [B, N, 1]

        # Normalizing to sum to 1. Z = probs.sum = 1 - rest.sum
        Z = probs.sum(dim=2, keepdim=True)                  # [B, 1, 1] 
        probs = probs / (Z + tol)                           # [B, N, 1]
        probs = probs.squeeze(-1)                           # [B, N]

        # Mapping
        probs = probs[:, self.mapping]                    # [B, N]
        
        dist = Categorical(probs=probs)

        return dist, (mean.squeeze(-1), std.squeeze(-1))    # mean,std=[B]

# GNN-K
class GNN_K_MLP(PolicyNetwork):
    def __init__(self,
                 input_size,
                 hidden_layers,
                 intervals, # list of size env.action_space.n (N) containing pairs (x_from, x_to)
                 mapping,   # list of size env.action_space.n (N) containing unique integers from 0 to N-1
                 K,         # num components
                 name = '',):
        super(GNN_K_MLP, self).__init__()

        self.K = K
        self.input_size = input_size
        self.output_size = 3*K            # (weight, mean, variance) for each of K components
        self._name = name

        self.register_buffer("intervals", torch.tensor(intervals, dtype=torch.float32, device=device).unsqueeze_(0))   # [1,N,2] to be transformed to [B,N,2]
        self.register_buffer("mapping", torch.tensor(mapping, dtype=torch.long, device=device))          # (N)

        self.xmin = min(intervals, key=lambda x : x[0])[0]     # leftmost in the interval
        self.xmax = max(intervals, key=lambda x : x[1])[1]     # rightmost in the interval


        # Network that outputs 2 values: mean and std
        layers = []
        last_size = self.input_size
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(last_size, hidden_size))
            layers.append(nn.LeakyReLU())
            last_size = hidden_size
        layers.append(nn.Linear(last_size, self.output_size))
        self.network = nn.Sequential(*layers).to(device)

    def forward(self, observation: torch.Tensor):
        
        B, O = observation.shape                                        # [B, O]

        out = self.network(observation)                                 # [B, 3*K] -> 3*K= (0.μ, 1.σ, 2.w)*K

        means = out[:, :self.K].unsqueeze(1)                            # [B, 1, K]

        stds = out[:, self.K:2*self.K].unsqueeze(1)                     # [B, 1, K]
        stds = fn.softplus(stds) + tol                                  # [B, 1, K]  softplus for numerical stability
        stds.clamp_min_(tol)
        ws = out[:, 2*self.K:3*self.K].unsqueeze(1)                     # [B, 1, K]
        ws = fn.softmax(ws, dim=-1)                                     # softmax weights on K dim to ensure > 0 and sum to 1
        # vars = stds**2                                                # [B, 1, K]

        intervals = self.intervals.expand(B, -1, -1)                    # [B, N, 2] -> batch_dim, num_envs, act_dim, (x_from, x_to).shape=2

        x_from = intervals[:, :,0].unsqueeze(-1)                        # [B, N, 1]
        x_to   = intervals[:, :,1].unsqueeze(-1)                        # [B, N, 1]

        # Integrating over intervals of interest
        probs = gaussian_mixture_integral(means, stds, ws, x_from=x_from, x_to=x_to)  # [B, N, 1]

        # Normalizing to sum to 1. Z = probs.sum = 1 - rest.sum
        Z = probs.sum(dim=2, keepdim=True)                                  # [B, 1, 1] 
        probs = probs / (Z + tol)                                           # [B, N, 1]
        probs = probs.squeeze(-1)                                           # [B, N]

        # Mapping
        probs = probs[:, self.mapping]                                    # [B, N]

        dist = Categorical(probs=probs)

        return dist, (means.squeeze(1), stds.squeeze(1), ws.squeeze(1))     # means,stds,ws=[B, K] (squeeze N dim)

# GNN-N
class GNN_N_MLP(PolicyNetwork):
    def __init__(self,
                 input_size,
                 hidden_layers,
                 N,         # action space size (N)
                 name = '',):
        super(GNN_N_MLP, self).__init__()

        self.N = N
        self.input_size = input_size
        self.output_size = 2*N            # (mean, variance) for each of N actions
        self._name = name

        # Network that outputs 2 values: mean and std
        layers = []
        last_size = self.input_size
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(last_size, hidden_size))
            layers.append(nn.LeakyReLU())
            last_size = hidden_size
        layers.append(nn.Linear(last_size, self.output_size))
        self.network = nn.Sequential(*layers).to(device)

    def forward(self, observation: torch.Tensor):
        
        B, E, O = observation.shape                                     # [B, E, O]

        # for name, p in self.named_parameters():
        #     if torch.isnan(p).any() or torch.isinf(p).any():
        #         print(f"PARAM NaN/Inf at {name}")

        out = self.network(observation)                                 # [B, E, 2*N] -> 2*N= (0.μ, 1.σ)*N

        means = out[:,:,:self.N]                                        # [B, E, N]

        stds = out[:,:,self.N:2*self.N]                                 # [B, E, N]

        stds = fn.softplus(stds) + tol                                  # [B, E, N]  softplus for numerical stability
        stds = stds.clamp_min(tol)
        # vars = stds**2                                                # [B, E, N]
        normal = torch.distributions.Normal(means, stds)
        # sample a z-score from each gaussian for each action (N) using the reparametrization trick
        samples = normal.rsample()
        samples = means + 0.05*(samples-means)                          # [B, E, N]     # 0.3 SHOULD BE NOISE COEF

        logits = samples.clamp(-20.0, 20.0)

        dist = Categorical(logits=logits)

        return dist, (means, stds)                                      # [B, E, N]


########### CNNs ############
# when observation is an image (other envs) - implement later (maybe Atari if it's not too complex, it seems ideal for pseudo-ordered actions)

# everything is the same as above except from O which will be H x W (or W x H). The appropriate Transformations should be applied
# e.g. resizing, cropping, augmentations, and then passed through the cnn (self.network with different architecture)

# forward stays the same after passing the input (O = image) through the self.network, because it should predict the same values.

# Logits CNN
class LogitsCNN(PolicyNetwork):
    def __init__(self, output_size: int, input_channels: int, input_height: int, input_width: int, conv_layers: list, fc_layers: list):
        super(LogitsCNN, self).__init__()

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

        self.head = nn.Linear(in_features, output_size)     # output_size = N (action space size)

        # Initialization
        # nn.init.constant_(self.head.bias, 0.0)
        # nn.init.orthogonal_(self.head.weight, gain=1.412)

        # print("Trainable parameters:")
        # for name, param in self.named_parameters():
        #     if param.requires_grad:
        #         print(f"{name:30} shape={tuple(param.shape)}   grad={param.grad}")

    def forward(self, observation: torch.Tensor):
        
        # print(observation.shape)
        B, E, C, H, W = observation.shape
        # print('[SHAPE]',observation.shape)

        # print('[OBS]', observation)

        # obs0 = observation[0, 0]
        # obs1 = observation[0, 1]
        # print("Identical across envs?", torch.equal(obs0, obs1))

        x = observation.view(B*E, C, H, W)  # join batch and env dimensions, conv2d expects [B, C, H, W] input

        x = self.conv(x)
        x = x.view(x.size(0), -1)   # flattening
        x = self.fc(x)
        x = self.head(x)
        logits = x.view(B, E, -1)    # converting [B x E, rest] -> [B, E, rest] to match the rest of the implementation
        # print('[LOGITS]', logits[0])
        probs = Categorical(logits=logits)
        return probs, logits



# Will not experiment with image-based environments for these methods as they have not performed well enough on simpler environments.
# # GNN CNN
# class GNN_CNN(PolicyNetwork):
#     pass

# # GNN-K CNN
# class GNN_K_CNN(PolicyNetwork):
#     pass

# GNN-N CNN
class GNN_N_CNN(PolicyNetwork):
    def __init__(self, output_size: int, input_channels: int, input_height: int, input_width: int, conv_layers: list, fc_layers: list):
        super(GNN_N_CNN, self).__init__()

        self.N = output_size

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

        self.head = nn.Linear(in_features, 2*output_size)     # output_size = 2*N (action space size)

        # Initialization
        # nn.init.constant_(self.head.bias, 0.0)
        # nn.init.orthogonal_(self.head.weight, gain=1.0)

    def forward(self, observation: torch.Tensor):


        B, E, C, H, W = observation.shape                               # [B, E, C, H, W]
        
        x = observation.view(B*E, C, H, W)  # join batch and env dimensions, conv2d expects [B, C, H, W] input
        x = self.conv(x)
        x = x.view(x.size(0), -1)   # flattening
        x = self.fc(x)                                                
        out = self.head(x).view(B, E, -1)                               # [B, E, 2*N] -> 2*N= (0.μ, 1.σ)*N

        means = out[:,:,:self.N]                                        # [B, E, N]

        stds = out[:,:,self.N:2*self.N]                                 # [B, E, N]

        stds = fn.softplus(stds) + tol                                  # [B, E, N]  softplus for numerical stability
        stds = stds.clamp_min(tol)
        # vars = stds**2                                                # [B, E, N]
        normal = torch.distributions.Normal(means, stds)
        # sample a z-score from each gaussian for each action (N) using the reparametrization trick
        samples = normal.rsample()
        samples = means + 0.05*(samples-means)                          # [B, E, N]     0.05 is the strength of the noise.

        logits = samples.clamp(-20.0, 20.0)

        dist = Categorical(logits=logits)

        return dist, (means, stds)                                      # [B, E, N]
