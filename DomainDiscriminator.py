import torch.nn as nn



class DomainDiscriminator(nn.Sequential):
    def __init__(self, in_feature: int, hidden_size: int, batch_norm=True, sigmoid=True):
        if sigmoid:
            final_layer = nn.Sequential(
                nn.Linear(hidden_size, 1),
                nn.Sigmoid()
            )
        else:
            final_layer = nn.Linear(hidden_size, 2)
        if batch_norm:
            super(DomainDiscriminator, self).__init__(
                nn.Linear(in_feature, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                final_layer
            )
        else:
            super(DomainDiscriminator, self).__init__(
                nn.Linear(in_feature, hidden_size),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                final_layer
            )

    def get_parameters(self):
        return [{"params": self.parameters(), "lr_mult": 1}]



