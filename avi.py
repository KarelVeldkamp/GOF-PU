import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.utils.prune
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset


class Encoder(pl.LightningModule):
    """
    Neural network used as encoder
    """
    def __init__(self,
                 nitems: int,
                 latent_dims: int,
                 hidden_layer_size: int):
        """
        Initialisation
        :param latent_dims: number of latent dimensions of the model
        """
        super(Encoder, self).__init__()

        input_layer = nitems

        self.dense1 = nn.Linear(input_layer, hidden_layer_size)
        self.densem = nn.Linear(hidden_layer_size, latent_dims)
        self.denses = nn.Linear(hidden_layer_size, latent_dims)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        A forward pass though the encoder network
        :param x: a tensor representing a batch of response data
        :param m: a mask representing which data is missing
        :return: a sample from the latent dimensions
        """

        # calculate s and mu based on encoder weights

        out = F.elu(self.dense1(x))
        mu =  self.densem(out)
        log_sigma = self.denses(out)
        #sigma = F.softplus(log_sigma)
        return mu, log_sigma


class SamplingLayer(pl.LightningModule):
    """
    class that samples from the approximate posterior using the reparametrisation trick
    """
    def __init__(self):
        super(SamplingLayer, self).__init__()
        self.N = torch.distributions.Normal(0, 1)

    def forward(self, mu, sigma):
        error = self.N.sample(mu.shape)
        # potentially move error vector to GPU
        error = error.to(mu)
        return mu + sigma.exp() * error


class Decoder(pl.LightningModule):
    """
    Neural network used as decoder
    """

    def __init__(self, nitems: int, latent_dims: int, qm: torch.Tensor=None):
        """
        Initialisation
        :param latent_dims: the number of latent factors
        :param qm: IxD Q-matrix specifying which items i<I load on which dimensions d<D
        """
        super().__init__()

        input_layer = latent_dims
        self.weights = nn.Parameter(torch.zeros((input_layer, nitems)))  # Manually created weight matrix
        self.bias = nn.Parameter(torch.zeros(nitems))  # Manually created bias vector
        self.activation = nn.Sigmoid()
        if qm is None:
            self.qm = torch.ones((latent_dims, nitems))
        else:
            self.qm = torch.Tensor(qm).t()

    def forward(self, x: torch.Tensor):
        self.qm = self.qm.to(self.weights)
        pruned_weights = self.weights * self.qm
        out = torch.matmul(x, pruned_weights) + self.bias
        out = self.activation(out)

        return out


class VAE(pl.LightningModule):
    """
    Neural network for the entire variational autoencoder
    """
    def __init__(self,
                 dataloader,
                 nitems: int,
                 latent_dims: int,
                 hidden_layer_size: int,
                 qm: torch.Tensor,
                 learning_rate: float,
                 n_samples: int = 1
                 ):
        """
        Initialisaiton
        :param latent_dims: number of latent dimensions
        :param qm: IxD Q-matrix specifying which items i<I load on which dimensions d<D
        """
        super(VAE, self).__init__()
        #self.automatic_optimization = False
        self.nitems = nitems
        self.latent_dims = latent_dims
        self.hidden_layer_size = hidden_layer_size

        self.dataloader = dataloader

        self.encoder = Encoder(nitems,
                               latent_dims,
                               hidden_layer_size
        )

        self.sampler = SamplingLayer()

        self.transform = nn.Identity()

        self.decoder = Decoder(nitems, latent_dims, qm)

        self.lr = learning_rate
        self.kl = 0
        self.n_samples = n_samples

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        """
        forward pass though the entire network
        :param x: tensor representing response data
        :param m: mask representing which data is missing
        :return: tensor representing a reconstruction of the input response data
        """
        mu, sigma = self.encoder(x)
        # reshape mu and log sigma in order to take multiple samples


        mu = mu.repeat(self.n_samples, 1, 1)
        sigma = sigma.repeat(self.n_samples, 1, 1)
        z = self.sampler(mu, sigma)
        z_tranformed = self.transform(z)
        reco = self.decoder(z_tranformed)
        return reco, mu, sigma, z


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, amsgrad=True)

    def training_step(self, batch, batch_idx):
        # forward pass

        data, mask = batch
        reco, mu, sigma, z = self(data, mask)

        mask = torch.ones_like(data)
        loss, _ = self.loss(data, reco, mask, mu, sigma, z)

        self.log('train_loss',loss)

        return {'loss': loss}

    def train_dataloader(self):
        return self.dataloader

    def loss(self, input, reco, mask, mu, sigma, z):
        #calculate log likelihood
        input = input.unsqueeze(0).repeat(reco.shape[0], 1, 1) # repeat input k times (to match reco size)
        log_p_x_theta = ((input * reco).clamp(1e-7).log() + ((1 - input) * (1 - reco)).clamp(1e-7).log()) # compute log ll
        logll = (log_p_x_theta * mask).sum(dim=-1, keepdim=True) # set elements based on missing data to zero
        #
        # calculate KL divergence
        log_q_theta_x = torch.distributions.Normal(mu, sigma.exp()).log_prob(z).sum(dim = -1, keepdim = True) # log q(Theta|X)
        log_p_theta = torch.distributions.Normal(torch.zeros_like(z).to(input), scale=torch.ones(mu.shape[2]).to(input)).log_prob(z).sum(dim = -1, keepdim = True) # log p(Theta)
        kl =  log_q_theta_x - log_p_theta # kl divergence

        # combine into ELBO
        elbo = logll - kl
        # # perform importance weighting
        with torch.no_grad():
            weight = (elbo - elbo.logsumexp(dim=0)).exp()
        #
        loss = (-weight * elbo).sum(0).mean()


        return loss, weight

    def fscores(self, batch, model, n_mc_samples=50):
        data, mask = batch

        if self.n_samples == 1:
            if model == 'cvae':
                mu, _ = self.encoder(data, mask)
            else:
                mu, _ = self.encoder(data)
            return mu.unsqueeze(0)
        else:
            scores = torch.empty((n_mc_samples, data.shape[0], self.latent_dims))
            for i in range(n_mc_samples):
                if model == 'cvae':
                    reco, mu, sigma, z = self(data, mask)
                else:
                    reco, mu, sigma, z = self(data, mask)

                loss, weight = self.loss(data, reco, mask, mu, sigma, z)

                idxs = torch.distributions.Categorical(probs=weight.permute(1,2,0)).sample()

                # Reshape idxs to match the dimensions required by gather
                # Ensure idxs is of the correct type
                idxs = idxs.long()

                # Expand idxs to match the dimensions required for gather
                idxs_expanded = idxs.unsqueeze(-1).expand(-1, -1, z.size(2))  # Shape [10000, 1, 3]

                # Use gather to select the appropriate elements from z
                output = torch.gather(z.transpose(0, 1), 1, idxs_expanded).squeeze().detach() # Shape [10000, 3]
                scores[i, :, :] = output

            return scores


class SimDataset(Dataset):
    """
    Torch dataset for item response data in numpy array
    """
    def __init__(self, X, device='cpu'):
        """
        initialize
        :param file_name: path to csv that contains NXI matrix of responses from N people to I items
        """
        # Read csv and ignore rownames


        self.x_train = torch.tensor(X, dtype=torch.float32)
        missing = torch.isnan(self.x_train)
        self.x_train[missing] = 0
        self.mask = (~missing).int()
        self.x_train = self.x_train.to(device)
        self.mask = self.mask.to(device)

    def __len__(self):
        return len(self.x_train)

    def __getitem__(self, idx):
        return self.x_train[idx], self.mask[idx]