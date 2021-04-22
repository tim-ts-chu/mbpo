
import torch
import numpy as np
from torch.nn import functional as F
from collections import OrderedDict
from mbpo.utils.logging import Progress, Silent

def construct_torch_disc(obs_dim=11, act_dim=3, rew_dim=1, hidden_dim=200, num_networks=7, num_elites=5, session=None):

    return Discriminator(obs_dim * 2 + rew_dim + act_dim, hidden_dim)

class TensorStandardScaler:
    def __init__(self, dim, device):
        self._dim = dim
        self._mu = torch.zeros(dim, dtype=torch.float, device=device)
        self._std = torch.ones(dim, dtype=torch.float, device=device)

    def fit(self, inputs):
        self._mu = inputs.mean(dim=0)
        self._std = inputs.std(dim=0)
        self._std[self._std < 1e-12] = 1.0

    def transform(self, inputs):
        return (inputs - self._mu) / self._std

class Discriminator:

    def __init__(self, input_dim, hidden_dim):
        self._device_id = 0 # hardcode

        self._model = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Linear(hidden_dim, 1)
            ).to(self._device_id)

        print('Discriminator:\n', self._model)

        self._optim = torch.optim.Adam(self._model.parameters(), lr=0.0002, betas=(0.5, 0.999))

        # self._scaler = TensorStandardScaler(input_dim, self._device_id)

        self._lambda_gp = 10


    def compute_gradient_penalty(self, real_samples, fake_samples):
        """Calculates the gradient penalty loss for WGAN GP"""
        # Random weight term for interpolation between real and fake samples
        alpha = torch.rand(real_samples.size(0), 1, device=self._device_id)
        # Get random interpolation between real and fake samples
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        d_interpolates = self._model(interpolates)
        #fake = torch.tensor(real_samples.shape[0], 1).fill_(1.0).to(device_id)
        fake = torch.full((real_samples.shape[0], 1), 1.0).to(self._device_id)#.fill_(1.0), requires_grad=False)
        # Get gradient w.r.t. interpolates
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    def train(self, inputs, targets, generator, latent_dim, output_dim, scaler):
        """
        @param inputs: (num_nets, batch_size, input_dim)
        @param targets: (num_nets, batch_size, output_dim)
        """

        # ignore dimension of num_nets since we don't do ensemble
        inputs = scaler.transform(inputs[0, :, :])
        targets = targets[0, :, :]

        # batch_size = inputs.shape[0]
        # noise = torch.randn((batch_size, latent_dim), device=self._device_id)
        # fake_targets = generator(torch.cat((inputs, noise), dim = 1))
        out = generator(inputs)
        mean, logvar = out[:, :output_dim], out[:, output_dim:]
        # fake_targets = mean + torch.randn_like(mean) * logvar.exp().sqrt()
        fake_targets = mean
        fake_batch = torch.cat((inputs, fake_targets), dim = 1)
        real_batch = torch.cat((inputs, targets), dim = 1)
        
        # Real images
        real_validity = self._model(real_batch)
        # Fake images
        fake_validity = self._model(fake_batch)
        
        gradient_penalty = self.compute_gradient_penalty(real_batch.data, fake_batch.data)
        disc_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + self._lambda_gp * gradient_penalty

        self._optim.zero_grad()
        disc_loss.backward()
        self._optim.step()
            
        disc_metrics = {'disc_loss': disc_loss.item()}
        return OrderedDict(disc_metrics)

    def validity(self, inputs, targets):
        batch = torch.cat((inputs, targets), dim = 1) 
        validity = self._model(batch)
        return validity

    def predict(self, inputs, ret_logits=False):
        """
        @param inputs: (batch, obs_dim + next_obs_diff_dim + rew_dim)
        @param ret_logits: if return logits, build computational graph; otherwise no_grad 

        Inputs here should NOT be scaled because the model discriminate on real data,
        and has its own scaler

        """

        if type(inputs) is not torch.Tensor:
            inputs = torch.from_numpy(inputs).to(self._device_id).float()

        if ret_logits:
            logits = self._model(self._scaler.transform(inputs))
            return logits
        else:
            with torch.no_grad():
                logits = self._model(self._scaler.transform(inputs))
            return torch.sigmoid(logits)


