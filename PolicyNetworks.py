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

        if observation.dim() == 1:
            observation = observation.unsqueeze(0)

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

        self.register_buffer("intervals", torch.tensor(intervals, dtype=torch.float32, device=device).unsqueeze_(0).unsqueeze_(0))   # [1,1,N,2] to be transformed to [B,E,N,2]
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
        
        B, E, O = observation.shape                     # [B, E, O]

        out = self.network(observation)                 # [B, E, 2] -> 2= 0.μ, 1.σ

        mean = out[:,:,0].unsqueeze(-1).unsqueeze(2)    # [B, E, 1, 1]
        # mean[..., -1] = tanh_squash_to_interval(mean[..., -1], self.xmin+tol, self.xmax-tol)    # ensure mean is not outside of mapping area.
        std = out[:,:,1].unsqueeze(-1).unsqueeze(2)     # [B, E, 1, 1]
        std = fn.softplus(std) + tol                    # [B, E, 1, 1] softplus for numerical stability
        std.clamp_min_(tol)
        # var = std**2                                  # [B, E, 1, 1]

        intervals = self.intervals.expand(B, E, -1, -1) # [B, E, N, 2]

        x_from = intervals[:,:,:,0].unsqueeze(-1)       # [B, E, N, 1]
        x_to   = intervals[:,:,:,1].unsqueeze(-1)       # [B, E, N, 1]

        # Integrating over intervals of interest
        probs = gaussian_integral(mean, std, x_from=x_from, x_to=x_to)  # [B, E, N, 1]

        # Normalizing to sum to 1. Z = probs.sum = 1 - rest.sum
        Z = probs.sum(dim=2, keepdim=True)                  # [B, E, 1, 1] 
        probs = probs / (Z + tol)                           # [B, E, N, 1]
        probs = probs.squeeze(-1)                           # [B, E, N]

        # Mapping
        probs = probs[:,:, self.mapping]                    # [B, E, N]
        
        dist = Categorical(probs=probs)

        return dist, (mean.squeeze(-1).squeeze(-1), std.squeeze(-1).squeeze(-1))    # [B, E]

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

        self.register_buffer("intervals", torch.tensor(intervals, dtype=torch.float32, device=device).unsqueeze_(0).unsqueeze_(0))   # [1,1,N,2] to be transformed to [B,E,N,2]
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
        
        B, E, O = observation.shape                                     # [B, E, O]

        out = self.network(observation)                                 # [B, E, 3*K] -> 3*K= (0.μ, 1.σ, 2.w)*K

        means = out[:,:,:self.K].unsqueeze(2)                           # [B, E, 1, K]
        # means[..., -1] = tanh_squash_to_interval(means[..., -1], self.xmin, self.xmax)    # ensure mean is not outside of mapping area.
        stds = out[:,:,self.K:2*self.K].unsqueeze(2)                    # [B, E, 1, K]
        stds = fn.softplus(stds) + tol                                  # [B, E, 1, K]  softplus for numerical stability
        stds.clamp_min_(tol)
        ws = out[:,:,2*self.K:3*self.K].unsqueeze(2)                    # [B, E, 1, K]
        ws = fn.softmax(ws, dim=3)                                      # softmax weights on K dim to ensure > 0 and sum to 1
        # vars = stds**2                                                  # [B, E, 1, K]

        intervals = self.intervals.expand(B, E, -1, -1)                 # [B, E, N, 2] -> batch_dim, num_envs, act_dim, (x_from, x_to).shape=2

        x_from = intervals[:,:,:,0].unsqueeze(-1)                       # [B, E, N, 1]
        x_to   = intervals[:,:,:,1].unsqueeze(-1)                       # [B, E, N, 1]

        # Integrating over intervals of interest
        probs = gaussian_mixture_integral(means, stds, ws, x_from=x_from, x_to=x_to)  # [B, E, N, 1]

        # Normalizing to sum to 1. Z = probs.sum = 1 - rest.sum
        Z = probs.sum(dim=2, keepdim=True)                                  # [B, E, 1, 1] 
        probs = probs / (Z + tol)                                           # [B, E, N, 1]
        probs = probs.squeeze(-1)                                           # [B, E, N]

        # Mapping
        probs = probs[:,:, self.mapping]                                    # [B, E, N]

        dist = Categorical(probs=probs)

        return dist, (means.squeeze(2), stds.squeeze(2), ws.squeeze(2))     # [B, E, K]

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
        # means[..., -1] = tanh_squash_to_interval(means[..., -1], self.xmin, self.xmax)    # ensure mean is not outside of mapping area.
        stds = out[:,:,self.N:2*self.N]                                 # [B, E, N]
        # print("first = ", stds)   # TODO remove it
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
    pass

# GNN CNN
class GNN_CNN(PolicyNetwork):
    pass

# GNN-K CNN
class GNN_K_CNN(PolicyNetwork):
    pass

# GNN-N CNN
class GNN_N_CNN(PolicyNetwork):
    pass