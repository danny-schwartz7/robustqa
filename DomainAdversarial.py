import torch
from torch import nn
import torch.nn.functional as F


# This is a classifier that predicts the 'domain' of an example given some subset of the QA model's
# internal state for that example. For now, it's an attention layer followed by a simple feed-forward NN.
class DomainDiscriminator(nn.Module):
    num_layers: int
    hidden_layers: nn.ModuleList
    attention: nn.MultiheadAttention
    query_param: nn.Parameter

    def __init__(self, num_domains: int, input_size: int, hidden_size: int = 768,
                 num_layers: int = 3, dropout: float = 0.1):
        super(DomainDiscriminator, self).__init__()
        self.num_layers = num_layers
        self.attention = nn.MultiheadAttention(hidden_size, 3)
        self.query_param = nn.Parameter(torch.empty((1, 1, hidden_size)))
        nn.init.xavier_uniform_(self.query_param)
        hidden_layers = []
        for i in range(num_layers):
            if i == 0:
                input_dim = input_size
            else:
                input_dim = hidden_size
            hidden_layers.append(nn.Sequential(
                nn.Linear(input_dim, hidden_size),
                nn.ReLU(), nn.Dropout(dropout)
            ))
        hidden_layers.append(nn.Linear(hidden_size, num_domains))
        self.hidden_layers = nn.ModuleList(hidden_layers)

    def forward(self, x):
        # forward pass
        attention_input = x.permute(1, 0, 2)
        query = torch.repeat_interleave(self.query_param, repeats=attention_input.shape[1], dim=1)
        # instead of just h_cls, use attention over all of the hidden states. The query is free to be optimized by the network.
        x, _ = self.attention(query=query, key=attention_input, value=attention_input, need_weights=False)
        x = torch.squeeze(x)
        for i in range(self.num_layers - 1):
            x = self.hidden_layers[i](x)
        logits = self.hidden_layers[-1](x)
        log_prob = F.log_softmax(logits, dim=1)
        return log_prob
