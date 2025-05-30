from Utilities import *
import torch.nn as nn

class PolicyNetwork(nn.Module):
    def __init__(self):
        super(PolicyNetwork, self).__init__()
        self.network = None # placeholder
    
    def train(self):
        if self.network is not None: self.network.train()

    def eval(self):
        if self.network is not None: self.network.eval()

    # Overriding dunder __call__ without changing its semantics, simply for the return types to be visible in the implicit forward call of model(.)  
    def __call__(self, *args, **kwargs) -> tuple[torch.distributions.categorical.Categorical, torch.Tensor]:
        return super().__call__(*args, **kwargs)

class ValueNetwork(nn.Module):
    def __init__(self):
        super(ValueNetwork, self).__init__()
        self.network = None # placeholder
    
    def train(self):
        if self.network is not None: self.network.train()

    def eval(self):
        if self.network is not None: self.network.eval()

    # Overriding dunder __call__ without changing its semantics, simply for the return types to be visible in the implicit forward call of model(.)  
    def __call__(self, *args, **kwargs) -> torch.Tensor:
        return super().__call__(*args, **kwargs)
