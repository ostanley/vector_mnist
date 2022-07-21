import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    """Conditional GAN generator"""

    def __init__(
        self, latent_dim: int, ngf: int, labels_dim: int, features_dim: int
    ) -> None:
        """
        Args:
            latent_dim (int): The size of the input noise.
            ngf (int): The size of the features is the generator layers.
            label_dim (int): The size of the label vector for each sample.
            features_dim (int): The number of the data features based on
            the dataset.
        """
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.labels_dim = labels_dim
        self.features_dim = features_dim
        self.ngf = ngf
        self.model = nn.Sequential(
            nn.Linear(self.latent_dim + self.labels_dim, self.ngf),
            nn.ReLU(inplace=True),
            nn.Linear(ngf, features_dim),
            nn.Tanh(),
        )

    def forward(
        self, noise: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        if self.labels_dim > 1:
            labels = F.one_hot(labels.to(torch.int64), num_classes=self.labels_dim)
            
        else:
            labels = torch.unsqueeze(labels, -1)
        g_input = torch.cat((noise, labels), -1)
        #import pdb; pdb.set_trace()
        x = self.model(g_input.to(torch.float))
        return x


class Discriminator(nn.Module):
    """Conditional GAN discriminator"""

    def __init__(self, ndf: int, features_dim: int, labels_dim: int) -> None:
        """
        Args:
            ndf (int): The size of the features is the discriminator layers.
            features_dim (int): The number of the data features based on the
            dataset.
            labels_dim (int): The size of the label vector for each sample
        """
        super(Discriminator, self).__init__()
        self.labels_dim = labels_dim
        self.features_dim = features_dim
        self.ndf = ndf
        self.model = nn.Sequential(
            nn.Linear(self.features_dim + self.labels_dim, self.ndf),
            nn.ReLU(inplace=True),
            nn.Linear(ndf, 1),
            nn.Sigmoid(),
        )

    def forward(
        self, input: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        if self.labels_dim > 1:
            labels = F.one_hot(labels.to(torch.int64), num_classes=self.labels_dim)
        else:
            labels = torch.unsqueeze(labels, -1)
        input = input.view(input.size(0), -1)
        d_input = torch.cat((input, labels), -1)
        x = self.model(d_input)
        return x
