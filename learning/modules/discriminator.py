import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self,
                 latent_channel,
                 num_classes,
                 device,
                 shape=[1024, 512],
                 ):
        super(Discriminator, self).__init__()
        self.input_dim = latent_channel
        self.num_classes = num_classes
        self.device = device
        self.shape = shape

        discriminator_layers = []
        curr_in_dim = self.input_dim
        for hidden_dim in self.shape:
            discriminator_layers.append(nn.Linear(curr_in_dim, hidden_dim))
            discriminator_layers.append(nn.ReLU())
            curr_in_dim = hidden_dim
        discriminator_layers.append(nn.Linear(self.shape[-1], self.num_classes))
        self.architecture = nn.Sequential(*discriminator_layers).to(self.device)
        self.architecture.train()

    def forward(self, x):
        return self.architecture(x)
    