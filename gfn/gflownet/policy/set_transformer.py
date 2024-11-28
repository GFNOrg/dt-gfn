import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from gflownet.policy.base import Policy

"""
Set Transformer Policy based on the "Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks" paper [1]. 
The Set Transformer definition in our code was based on their implementation. 

[1]: @InProceedings{lee2019set,
    title={Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks},
    author={Lee, Juho and Lee, Yoonho and Kim, Jungtaek and Kosiorek, Adam and Choi, Seungjin and Teh, Yee Whye},
    booktitle={Proceedings of the 36th International Conference on Machine Learning},
    pages={3744--3753},
    year={2019}
}
"""


class MAB(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super(MAB, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        # Keep separate projections for stability
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

    def forward(self, Q, K):
        if Q.size(0) == 0:
            return Q

        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)

        A = torch.softmax(Q_.bmm(K_.transpose(1, 2)) / math.sqrt(self.dim_V), 2)
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = O if getattr(self, "ln0", None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, "ln1", None) is None else self.ln1(O)
        return O


class SAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, ln=False):
        super(SAB, self).__init__()
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(X, X)


class ISAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False):
        super(ISAB, self).__init__()
        self.I = nn.Parameter(torch.Tensor(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads, ln=ln)
        self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads, ln=ln)

    def forward(self, X):
        H = self.mab0(self.I.repeat(X.size(0), 1, 1), X)
        return self.mab1(X, H)


class PMA(nn.Module):
    def __init__(self, dim, num_heads, num_seeds, ln=False):
        super(PMA, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim, dim, dim, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(self.S.repeat(X.size(0), 1, 1), X)


class SetTransformer(nn.Module):
    def __init__(
        self,
        dim_input,
        num_outputs,
        dim_output,
        num_inds=16,
        dim_hidden=64,
        num_heads=2,
        ln=False,
    ):
        super(SetTransformer, self).__init__()
        self.enc = nn.ModuleList(
            [
                ISAB(dim_input, dim_hidden, num_heads, num_inds, ln=ln),
                ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln),
            ]
        )
        self.dec = nn.ModuleList(
            [
                PMA(dim_hidden, num_heads, num_outputs, ln=ln),
                SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
                SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
            ]
        )
        self.final_linear = nn.Linear(dim_hidden, dim_output)

    def forward(self, X):
        for layer in self.enc:
            X = layer(X)
        for layer in self.dec:
            X = layer(X)
        return self.final_linear(X)


class TreeSetTransformer(nn.Module):
    def __init__(
        self,
        backbone_args: dict,
        num_attributes=6,
        max_depth=4,
        embedding_dim=64,
    ):
        super(TreeSetTransformer, self).__init__()
        self.fc1 = nn.Linear(num_attributes * max_depth, embedding_dim)
        self.fc2 = nn.Linear(
            backbone_args["num_outputs"] * backbone_args["dim_output"],
            backbone_args["dim_output"],
        )
        self.model = SetTransformer(**backbone_args)
        assert embedding_dim == backbone_args["dim_input"]

    def forward(self, x):
        if x.size(0) == 0:
            return torch.zeros((0, self.fc2.out_features), device=x.device)
        batch = self.fc1(x)
        batch = self.model(batch)
        batch = batch.flatten(1)
        batch = self.fc2(batch)
        return batch


class TreeSetPolicy(Policy):
    def __init__(
        self,
        config,
        env,
        device,
        float_precision,
        base=None,
    ):
        self.env = env
        self.policy_output_dim = env.policy_output_dim
        self.backbone_args = {
            "dim_input": 64,
            "num_outputs": 1,
            "dim_output": self.policy_output_dim,
        }
        self.policy_args = {
            "num_attributes": 6,
            "max_depth": env.max_depth,
            "embedding_dim": 64,
        }
        self.checkpoint = None

        super().__init__(
            config=config,
            env=env,
            device=device,
            float_precision=float_precision,
            base=base,
        )

        self.is_model = True

    def parse_config(self, config):
        if config is not None:
            self.backbone_args.update(config.get("backbone_args", {}))
            self.policy_args.update(config.get("policy_args", {}))
        assert self.policy_output_dim == self.env.policy_output_dim

    def instantiate(self):
        self.model = TreeSetTransformer(
            backbone_args=self.backbone_args, **self.policy_args
        ).to(self.device)

        if self.device == "cuda":
            self.model = torch.compile(self.model, mode="default", fullgraph=False)

    def __call__(self, states):
        if states.dim() == 2:
            states = states.unsqueeze(0)
        return self.model(states)
