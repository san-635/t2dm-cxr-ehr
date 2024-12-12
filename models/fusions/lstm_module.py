import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, args, input_dim, batch_first=True):
        super(LSTM, self).__init__()
        self.layers = args.layers
        for layer in range(args.layers):
            setattr(self, f'layer{layer}', nn.LSTM(input_dim, args.dim, batch_first=batch_first, dropout=args.dropout))
            input_dim = args.dim
        self.dropout = None
        if args.dropout > 0.0:
            self.dropout = nn.Dropout(args.dropout)
        self.feats_dim = args.dim
        self.classifier = nn.Linear(args.dim, args.num_classes)
        self.initialize_weights()

    def initialize_weights(self):
        for module in self.modules():
            if type(module) in [nn.Linear]:
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
            elif type(module) in [nn.LSTM, nn.RNN, nn.GRU]:
                # _hh_: hidden-hidden, _ih_: input-hidden
                nn.init.orthogonal_(module.weight_hh_l0)
                nn.init.xavier_uniform_(module.weight_ih_l0)
                nn.init.zeros_(module.bias_hh_l0)
                nn.init.zeros_(module.bias_ih_l0)

    def forward(self, x, seq_lengths):
        x = torch.nn.utils.rnn.pack_padded_sequence(x, seq_lengths, batch_first=True, enforce_sorted=False)
        for layer in range(self.layers):
            x, (ht, _) = getattr(self, f'layer{layer}')(x)
        feats = ht[-1]
        if self.dropout is not None:
            feats = self.dropout(feats)
        out = self.classifier(feats)
        preds = torch.sigmoid(out)
        return preds, feats                                 # return logits (for encoder pre-training) and feature representation (for model training)