import torch
import torch.nn as nn
import math

from transformers.models.bert.modeling_bert import BertPredictionHeadTransform

class PositionalEncoding(nn.Module):
    def __init__(self, hidden_size, dropout=0.1, max_len=5000):             # any max_len > max_text_len and max_ecg_len is sufficient
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, hidden_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_size, 2).float() * (-math.log(10000.0) / hidden_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1), :]                                      # since x.size(1) = max_text_len
        return self.dropout(x)

class EHREmbedding(nn.Module):
    def __init__(self, ehr_n_var, hidden_size):
        super().__init__()
        self.feature_embedding = nn.Linear(ehr_n_var, hidden_size)
        self.positional_encoding = PositionalEncoding(hidden_size)

    def forward(self, x):
        x = x.to(self.feature_embedding.weight.device)
        x = self.feature_embedding(x)

        cls_token = torch.zeros(x.size(0), 1, x.size(2), device=x.device)   # the extra learnable [class] token for EHR modality
        x = torch.cat([cls_token, x], dim=1)                                # prepend [class] token to x, along the no. of rows
        x = self.positional_encoding(x)                                     # element-wise addition of position embeddings
        return x

class ECGEmbedding(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.feature_embedding = nn.Linear(12, hidden_size)
        self.positional_encoding = PositionalEncoding(hidden_size)

    def forward(self, x):
        x = x.to(self.feature_embedding.weight.device)
        x = self.feature_embedding(x)

        cls_token = torch.zeros(x.size(0), 1, x.size(2), device=x.device)   # the extra learnable [class] token for ECG modality
        x = torch.cat([cls_token, x], dim=1)                                # prepend [class] token to x, along the no. of rows
        x = self.positional_encoding(x)                                     # element-wise addition of position embeddings
        return x

class Pooler(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, 0]                            # extract the [class] token
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class ITMHead(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, x):
        x = self.fc(x)
        return x


class MLMHead(nn.Module):
    def __init__(self, config, weight=None):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        if weight is not None:
            self.decoder.weight = weight

    def forward(self, x):
        x = self.transform(x)
        x = self.decoder(x) + self.bias
        return x


class MPPHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)
        self.decoder = nn.Linear(config.hidden_size, 256 * 3)

    def forward(self, x):
        x = self.transform(x)
        x = self.decoder(x)
        return x
