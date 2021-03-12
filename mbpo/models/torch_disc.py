
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
            torch.nn.LeakyReLU(),
            # torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.LeakyReLU(),
            # torch.nn.ReLU(),
            # torch.nn.Linear(hidden_dim, hidden_dim),
            # torch.nn.LeakyReLU(),
            # torch.nn.Linear(hidden_dim, hidden_dim),
            # torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_dim, 1)
            ).to(self._device_id)

        self._optim = torch.optim.AdamW(self._model.parameters(), lr=0.001)

        self._scaler = TensorStandardScaler(input_dim, self._device_id)

    def train(self, inputs, targets, disc_batch_size):

        holdout_ratio = 0.1 # hardcode

        # Split into training and holdout sets
        num_holdout = int(inputs.shape[0] * holdout_ratio)
        permutation = np.random.permutation(inputs.shape[0])
        inputs, holdout_inputs = inputs[permutation[num_holdout:]], inputs[permutation[:num_holdout]] # (57139, 23)
        targets, holdout_targets = targets[permutation[num_holdout:]], targets[permutation[:num_holdout]] # (5000, 23)

        inputs, holdout_inputs = torch.from_numpy(inputs).to(self._device_id), torch.from_numpy(holdout_inputs).to(self._device_id)
        targets, holdout_targets = torch.from_numpy(targets).to(self._device_id), torch.from_numpy(holdout_targets).to(self._device_id)

        self._scaler.fit(inputs)
        idxs = np.random.randint(inputs.shape[0], size=inputs.shape[0])
        progress = Progress(int(np.ceil(idxs.shape[-1] / disc_batch_size)))
        for batch_num in range(int(np.ceil(idxs.shape[-1] / disc_batch_size))):
            batch_idxs = idxs[batch_num * disc_batch_size:(batch_num + 1) * disc_batch_size]

            logits = self._model(self._scaler.transform(inputs[batch_idxs, :])).flatten()
            loss = F.binary_cross_entropy_with_logits(logits, targets[batch_idxs].flatten())

            self._optim.zero_grad()
            loss.backward()
            self._optim.step()

            progress.set_description([['disc_loss', loss]])
            progress.update()
            
        with torch.no_grad():
            logits = self._model(self._scaler.transform(holdout_inputs)).flatten()
            val_loss = F.binary_cross_entropy_with_logits(logits, holdout_targets.flatten())

        disc_metrics = {'disc_val_loss': val_loss, 'disc_loss': loss}
        print('disc_val_loss:', val_loss)
        return OrderedDict(disc_metrics)

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


