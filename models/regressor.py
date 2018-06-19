import torch.nn as nn


class Regressor(nn.Module):
    """
    For regression, takes in a feature extractor (E.g. DanQ) and regresses with dense layers.
    """
    def __init__(self, input_size, out_size, feature_extractor):
        super(Regressor, self).__init__()

        if input_size // 2 > 512:
            hidden_size = 512

        else:
            hidden_size = input_size // 2

        self.feature_extractor = feature_extractor
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, out_size)
        )

    def forward(self, x):
        x = self.feature_extractor(x)

        if isinstance(x, tuple):
            x = x[0]

        return self.model(x.view(x.size(0), -1))  # Flatten then run the linear layer.

    def weight_func(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            m.bias.data.zero_()

    def initialize_weights(self):
        self.feature_extractor.initialize_weights()
        self.model.apply(self.weight_func)


