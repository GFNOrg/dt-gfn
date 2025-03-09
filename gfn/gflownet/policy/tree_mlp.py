import torch
import torch.nn as nn
from gflownet.policy.base import Policy

class MLPPolicy(Policy):
    def __init__(self, config, env, float_precision=32, device='cuda', base=None):
        super().__init__(config, env, float_precision, device, base)
        self.sigma = nn.Parameter(torch.tensor(0.5))  # Initialize sigma
        self.phi = nn.Parameter(torch.tensor(0.5))    # Initialize phi
        self.instantiate()

    def instantiate(self):
        if self.type == "mlp":
            self.model = self.make_mlp(nn.LeakyReLU()).to(self.device)
            if self.device == 'cuda': 
               self.model = torch.compile(self.model)
            self.is_model = True
        else:
            super().instantiate()

    def forward(self, states):
        return super().__call__(states)
