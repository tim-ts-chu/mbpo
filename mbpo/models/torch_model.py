
import time
import torch
import itertools
import numpy as np
from torch.nn import functional as F
from collections import OrderedDict
from mbpo.utils.logging import Progress, Silent

def construct_torch_model(obs_dim=11, act_dim=3, rew_dim=1, hidden_dim=200, num_networks=7, num_elites=5, session=None):

    model = WorldModel(
            num_networks,
            num_elites,
            input_dim=obs_dim+act_dim+act_dim, # second act_dim is latent dim for gan
            # input_dim=obs_dim+act_dim,
            output_dim=obs_dim+rew_dim,
            hidden_dim=hidden_dim,
            latent_dim=act_dim)

    return model

class TensorStandardScaler:
    def __init__(self, dim, device, norm_type):
        self._dim = dim
        self._mu = torch.zeros(dim, dtype=torch.float, device=device)
        self._std = torch.ones(dim, dtype=torch.float, device=device)
        self._norm_type = norm_type # can normalize data by either 'std' or 'minmax'

    def fit(self, inputs):
        self._mu = inputs.mean(dim=0)
        self._std = inputs.std(dim=0)
        self._min, _ = inputs.min(dim=0)
        self._max, _ = inputs.max(dim=0)
        self._std[self._std < 1e-12] = 1.0

    def transform(self, inputs):
        if self._norm_type == 'std':
            return (inputs - self._mu) / self._std
        elif self._norm_type == 'minmax':
            return 2 * ((inputs - self._min) / (self._max - self._min)) - 1
        else:
            raise TypeError('wrong norm_type: {}'.format(self._norm_type))

    def inverse(self, inputs):
        if self._norm_type == 'std':
            return inputs * (2 * self._std) + self._mu
        elif self._norm_type == 'minmax':
            return ((inputs + 1) / 2) * (self._max - self._min) + self._min
        else:
            raise TypeError('wrong norm_type: {}'.format(self._norm_type))

class WorldNet(torch.nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(WorldNet, self).__init__()
        self.fc_in = torch.nn.Linear(input_dim, hidden_dim)
        self.fc_1 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc_2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc_3 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = torch.nn.Linear(hidden_dim, output_dim*2)

        den = 2 * torch.tensor(input_dim).float().sqrt()
        torch.nn.init.normal_(self.fc_in.weight, std=1.0 / den)
        torch.nn.init.normal_(self.fc_1.weight, std=1.0 / den)
        torch.nn.init.normal_(self.fc_2.weight, std=1.0 / den)
        torch.nn.init.normal_(self.fc_3.weight, std=1.0 / den)
        torch.nn.init.normal_(self.fc_out.weight, std=1.0 / den)

    def forward(self, x):
        out = self.swish(self.fc_in(x))
        out = self.swish(self.fc_1(out))
        out = self.swish(self.fc_2(out))
        out = self.swish(self.fc_3(out))
        out = self.fc_out(out)
        return out

    def swish(self, x):
        return x * torch.sigmoid(x)

    def get_decays(self):
        decays =  0.000025 * (self.fc_in.weight ** 2).sum() / 2.0
        decays += 0.000050 * (self.fc_1.weight ** 2).sum() / 2.0
        decays += 0.000075 * (self.fc_2.weight ** 2).sum() / 2.0
        decays += 0.000075 * (self.fc_3.weight ** 2).sum() / 2.0
        decays += 0.0001 * (self.fc_out.weight ** 2).sum() / 2.0
        return decays

class WorldModel:

    def __init__(self, num_networks, num_elites, input_dim, output_dim, hidden_dim, latent_dim):

        self._device_id = 0 # hardcode
        self.num_nets = num_networks
        self.num_elites = num_elites
        self._latent_dim = latent_dim
        self._output_dim = output_dim
        self._model = {}
        self._scaler = TensorStandardScaler(input_dim, self._device_id, 'std')
        self._num_samples = None
        
        self._parameters = []
        for i in range(num_networks):
            self._model[i] = WorldNet(input_dim, hidden_dim, output_dim).to(self._device_id)
            print(self._model[i])
            self._parameters += list(self._model[i].parameters())

        self._max_logvar = torch.full((output_dim,), 0.5, requires_grad=True, device=self._device_id, dtype=torch.float)
        self._min_logvar = torch.full((output_dim,), -10, requires_grad=True, device=self._device_id, dtype=torch.float)

        self._optim = torch.optim.Adam([
            {'params': self._model[0].fc_in.parameters()},
            {'params': self._model[0].fc_1.parameters()},
            {'params': self._model[0].fc_2.parameters()},
            {'params': self._model[0].fc_3.parameters()},
            {'params': self._model[0].fc_out.parameters()},
            {'params': [self._max_logvar, self._min_logvar]}
            ], lr=0.001)
            # ], lr=0.0002, betas=(0.5, 0.999))

        self._scheduler = torch.optim.lr_scheduler.MultiStepLR(self._optim, milestones=[10, 50], gamma=0.5)

    def _losses(self, inputs, targets, mse_only=False, disc=None, ret_prog=False):
        """
        inputs: (num_nets, batch_size, state_dim + act_dim)
        targets: (num_nets, batch_size, state_dim + rew_dim)
        ret: (num_nets, )
        """
        losses = []
        batch_size = inputs.shape[1]
        for i in range(self.num_nets):
            noise = torch.randn(batch_size, self._latent_dim, device=inputs.device)
            model_input = torch.cat((inputs[i,:,:], noise), dim = 1)
            # model_input = inputs[i,:,:]
            out = self._model[i](model_input)
            mean, logvar = out[:, :self._output_dim], out[:, self._output_dim:]

            logvar = self._max_logvar - F.softplus(self._max_logvar - logvar)
            logvar = self._min_logvar + F.softplus(logvar - self._min_logvar)

            mse_loss = F.mse_loss(mean, targets[i,:,:]) # scalar
            if mse_only:
                losses.append(mse_loss)
                continue

            mean_mse = mean.clone()
            mean_mse.retain_grad()
            inv_var = torch.exp(-logvar)
            train_loss = (((mean_mse - targets[i,:,:]) ** 2) * inv_var) + logvar
            train_loss = train_loss.mean(-1).mean(-1)

            # reg loss
            var_loss = 0.01 * (self._max_logvar.sum() - self._min_logvar.sum())
            reg_loss = self._model[i].get_decays()

            # gan loss
            mean_gan = mean.clone()
            mean_gan.retain_grad()
            gan_validity = disc.validity(inputs[i,:,:], mean_gan)
            gan_loss = -torch.mean(gan_validity)
            # gan_loss = torch.tensor(0)

            total_loss = train_loss + var_loss + reg_loss + 0.1 * gan_loss

            losses.append(total_loss)
            
            if ret_prog:
                prog = [
                    ['train', train_loss.item()],
                    ['var', var_loss.item()],
                    ['reg', reg_loss.item()],
                    ['mse', mse_loss.item()],
                    ['gan', gan_loss.item()]]
                info = {
                        'gan_loss': gan_loss.item(),
                        'mean_mse': mean_mse,
                        'mean_gan': mean_gan,
                        }
                return torch.stack(losses), prog, info

        return torch.stack(losses)


    def _start_train(self):
        self._state = {}
        self._snapshots = {i: (None, 1e10) for i in range(self.num_nets)}
        self._epochs_since_update = 0

    def _end_train(self, holdout_losses):
        sorted_inds = np.argsort(holdout_losses)
        self._model_inds = sorted_inds[:self.num_elites].tolist()
        print('Using {} / {} models: {}'.format(self.num_elites, self.num_nets, self._model_inds))

    def _save_best(self, epoch, holdout_losses):
        updated = False
        for i in range(len(holdout_losses)):
            current = holdout_losses[i]
            _, best = self._snapshots[i]
            improvement = (best - current) / best
            if improvement > 0.01:
                self._snapshots[i] = (epoch, current)
                # self._save_state(i)
                updated = True
                improvement = (best - current) / best
                # print('epoch {} | updated {} | improvement: {:.4f} | best: {:.4f} | current: {:.4f}'.format(epoch, i, improvement, best, current))

        if updated:
            self._epochs_since_update = 0
        else:
            self._epochs_since_update += 1

        if self._epochs_since_update > self._max_epochs_since_update:
            # print('[ BNN ] Breaking at epoch {}: {} epochs since update ({} max)'.format(epoch, self._epochs_since_update, self._max_epochs_since_update))
            return True
        else:
            return False

    def scheduler_step(self):
        self._scheduler.step()

    def train(self, inputs, targets, disc,
              batch_size=32, max_epochs=None, max_epochs_since_update=5,
              hide_progress=False, holdout_ratio=0.0, max_logging=5000, max_grad_updates=None, timer=None, max_t=None, num_samples=None):
        """Trains/Continues network training

        Arguments:
            inputs (np.ndarray): Network inputs in the training dataset in rows.
            targets (np.ndarray): Network target outputs in the training dataset in rows corresponding
                to the rows in inputs.
            batch_size (int): The minibatch size to be used for training.
            epochs (int): Number of epochs (full network passes that will be done.
            hide_progress (bool): If True, hides the progress bar shown at the beginning of training.

        Returns: None
        """
        self._max_epochs_since_update = max_epochs_since_update
        self._start_train()
        break_train = False

        def shuffle_rows(arr):
            idxs = np.argsort(np.random.uniform(size=arr.shape), axis=-1)
            return arr[np.arange(arr.shape[0])[:, None], idxs]

        # Split into training and holdout sets
        num_holdout = min(int(inputs.shape[0] * holdout_ratio), max_logging)
        permutation = np.random.permutation(inputs.shape[0])
        inputs, holdout_inputs = inputs[permutation[num_holdout:]], inputs[permutation[:num_holdout]] # (57139, 23)
        targets, holdout_targets = targets[permutation[num_holdout:]], targets[permutation[:num_holdout]] # (5000, 23)
        holdout_inputs = np.tile(holdout_inputs[None], [self.num_nets, 1, 1])
        holdout_targets = np.tile(holdout_targets[None], [self.num_nets, 1, 1])

        inputs, holdout_inputs = torch.from_numpy(inputs).to(self._device_id), torch.from_numpy(holdout_inputs).to(self._device_id)
        targets, holdout_targets = torch.from_numpy(targets).to(self._device_id), torch.from_numpy(holdout_targets).to(self._device_id)

        print('[ BNN ] Training {} | Holdout: {}'.format(inputs.shape, holdout_inputs.shape)) #[ BNN ] Training (57139, 23) | Holdout: (7, 5000, 23)
        self._scaler.fit(inputs)
        inputs = self._scaler.transform(inputs)
        holdout_inputs = self._scaler.transform(holdout_inputs)

        idxs = np.random.randint(inputs.shape[0], size=[self.num_nets, inputs.shape[0]])
        if hide_progress:
            progress = Silent()
        else:
            progress = Progress(max_epochs)

        if max_epochs:
            epoch_iter = range(max_epochs)
        else:
            epoch_iter = itertools.count()

        # else:
        #     epoch_range = trange(epochs, unit="epoch(s)", desc="Network training")

        model_metrics = {}
        t0 = time.time()
        grad_updates = 0

        for epoch in epoch_iter:
            for batch_num in range(int(np.ceil(idxs.shape[-1] / batch_size))):
                batch_idxs = idxs[:, batch_num * batch_size:(batch_num + 1) * batch_size]

                # Train Discriminator
                disc_metrics = disc.train(inputs[batch_idxs, :], targets[batch_idxs, :], self._model[0], self._latent_dim, self._output_dim, self._scaler)
                model_metrics.update(disc_metrics)

                self._num_samples = num_samples 
                losses, prog, info = self._losses(inputs[batch_idxs, :], targets[batch_idxs, :], disc=disc, ret_prog=True)
                self._optim.zero_grad()
                losses.sum().backward()
                model_metrics.update({
                    'gan_loss': info['gan_loss'],
                    'gradnorm_mse_output': torch.nn.utils.clip_grad_norm_(info['mean_mse'], 10e9),
                    'gradnorm_gan_output': torch.nn.utils.clip_grad_norm_(info['mean_gan'], 10e9),
                    'gradnorm_model_weight': torch.nn.utils.clip_grad_norm_(self._parameters, 10e9),
                    })
                self._optim.step()
                grad_updates += 1

            idxs = shuffle_rows(idxs)

            if not hide_progress:
                if holdout_ratio < 1e-12:
                    # losses = self.sess.run(
                            # self.mse_loss,
                            # feed_dict={
                                # self.sy_train_in: inputs[idxs[:, :max_logging]],
                                # self.sy_train_targ: targets[idxs[:, :max_logging]]
                            # }
                        # )
                    # TODO model loss
                    losses = self._losses(inputs[idxs[:, :max_logging], :], targets[idxs[:, :max_logging], :], mse_only=True)
                    losses = losses.cpu().detach().numpy()

                    named_losses = [['M{}'.format(i), losses[i]] for i in range(len(losses))]
                    progress.set_description(named_losses)
                else:
                    # losses = self.sess.run(
                            # self.mse_loss,
                            # feed_dict={
                                # self.sy_train_in: inputs[idxs[:, :max_logging]],
                                # self.sy_train_targ: targets[idxs[:, :max_logging]]
                            # }
                        # )
                    # holdout_losses = self.sess.run(
                            # self.mse_loss,
                            # feed_dict={
                                # self.sy_train_in: holdout_inputs,
                                # self.sy_train_targ: holdout_targets
                            # }
                        # )
                    # TODO model loss and holdout loss
                    losses = self._losses(inputs[idxs[:, :max_logging], :], targets[idxs[:, :max_logging], :], mse_only=True)
                    losses = losses.cpu().detach().numpy()
                    holdout_losses = self._losses(holdout_inputs, holdout_targets, mse_only=True)
                    holdout_losses = holdout_losses.cpu().detach().numpy()

                    named_losses = [['M{}'.format(i), losses[i]] for i in range(len(losses))]
                    named_holdout_losses = [['V{}'.format(i), holdout_losses[i]] for i in range(len(holdout_losses))]
                    named_losses = named_losses + named_holdout_losses + [['T', time.time() - t0]] + prog
                    progress.set_description(named_losses)

                    break_train = self._save_best(epoch, holdout_losses)

            progress.update()
            t = time.time() - t0
            if break_train or (max_grad_updates and grad_updates > max_grad_updates):
                break
            if max_t and t > max_t:
                descr = 'Breaking because of timeout: {}! (max: {})'.format(t, max_t)
                progress.append_description(descr)
                # print('Breaking because of timeout: {}! | (max: {})\n'.format(t, max_t))
                # time.sleep(5)
                break

        progress.stamp()
        if timer: timer.stamp('bnn_train')

        # self._set_state()
        if timer: timer.stamp('bnn_set_state')

        # TODO holdout_losses
        # holdout_losses = self.sess.run(
            # self.mse_loss,
            # feed_dict={
                # self.sy_train_in: holdout_inputs,
                # self.sy_train_targ: holdout_targets
            # }
        # )
        holdout_losses = self._losses(holdout_inputs, holdout_targets, mse_only=True)
        holdout_losses = holdout_losses.cpu().detach().numpy()

        if timer: timer.stamp('bnn_holdout')

        self._end_train(holdout_losses)
        if timer: timer.stamp('bnn_end')

        val_loss = (np.sort(holdout_losses)[:self.num_elites]).mean()
        model_metrics.update({'val_loss': val_loss})
        print('[ BNN ] Holdout (Torch)', np.sort(holdout_losses), model_metrics)
        return OrderedDict(model_metrics)


    def predict(self, inputs, factored=False, *args, **kwargs):
        """Returns the distribution predicted by the model for each input vector in inputs.
        Behavior is affected by the dimensionality of inputs and factored as follows:

        inputs is 2D, factored=True: Each row is treated as an input vector.
            Returns a mean of shape [ensemble_size, batch_size, output_dim] and variance of shape
            [ensemble_size, batch_size, output_dim], where N(mean[i, j, :], diag([i, j, :])) is the
            predicted output distribution by the ith model in the ensemble on input vector j.

        inputs is 2D, factored=False: Each row is treated as an input vector.
            Returns a mean of shape [batch_size, output_dim] and variance of shape
            [batch_size, output_dim], where aggregation is performed as described in the paper.

        inputs is 3D, factored=True/False: Each row in the last dimension is treated as an input vector.
            Returns a mean of shape [ensemble_size, batch_size, output_dim] and variance of sha
            [ensemble_size, batch_size, output_dim], where N(mean[i, j, :], diag([i, j, :])) is the
            predicted output distribution by the ith model in the ensemble on input vector [i, j].

        Arguments:
            inputs (np.ndarray): An array of input vectors in rows. See above for behavior.
            factored (bool): See above for behavior.
        """
        if len(inputs.shape) == 2:
            if factored:
                # return self.sess.run(
                    # [self.sy_pred_mean2d_fac, self.sy_pred_var2d_fac],
                    # feed_dict={self.sy_pred_in2d: inputs}
                # )

                batch_size = inputs.shape[0]
                x = torch.from_numpy(inputs).to(self._device_id).float()
                means = []
                varis = []
                with torch.no_grad():
                    for i in range(self.num_nets):
                        x = self._scaler.transform(x)
                        noise = torch.randn(batch_size, self._latent_dim, device=x.device)
                        out = self._model[i](torch.cat((x, noise), dim = 1))
                        # out = self._model[i](x)
                        means.append(out[:, :self._output_dim])
                        logvar = out[:, self._output_dim:]
                        logvar = self._max_logvar - F.softplus(self._max_logvar - logvar)
                        logvar = self._min_logvar + F.softplus(logvar - self._min_logvar)
                        varis.append(logvar.exp())

                return torch.stack(means).cpu().detach().numpy(), torch.stack(varis).cpu().detach().numpy()

            else:
                # return self.sess.run(
                    # [self.sy_pred_mean2d, self.sy_pred_var2d],
                    # feed_dict={self.sy_pred_in2d: inputs}
                # )
                raise NotImplementedError()
        else:
            # return self.sess.run(
                # [self.sy_pred_mean3d_fac, self.sy_pred_var3d_fac],
                # feed_dict={self.sy_pred_in3d: inputs}
            # )
            raise NotImplementedError()

    def random_inds(self, batch_size):
        inds = np.random.choice(self._model_inds, size=batch_size)
        return inds

    def create_prediction_tensors(self, inputs, factored=False, *args, **kwargs):
        """See predict() above for documentation.
        """
        factored_mean, factored_variance = self._compile_outputs(inputs)
        if inputs.shape.ndims == 2 and not factored:
            mean = tf.reduce_mean(factored_mean, axis=0)
            variance = tf.reduce_mean(tf.square(factored_mean - mean), axis=0) + \
                       tf.reduce_mean(factored_variance, axis=0)
            return mean, variance
        return factored_mean, factored_variance

