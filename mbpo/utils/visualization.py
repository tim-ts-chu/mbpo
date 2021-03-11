import io
import math
import torch
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pdb
from mbpo.utils.plotter import Plotter

def plot_trajectories(writer, label, epoch, env_traj, model_traj, means, stds, conf_r, conf_f):
    state_dim = env_traj[0].size
    model_states = [[obs[s] for obs in model_traj] for s in range(state_dim)]
    env_states   = [[obs[s] for obs in env_traj  ] for s in range(state_dim)]

    means = [np.array([mean[s] for mean in means]) for s in range(state_dim)]
    stds = [np.array([std[s] for std in stds]) for s in range(state_dim)]

    cols = 1
    # rows = math.ceil(state_dim / cols)
    rows = math.ceil(state_dim + 1 / cols)

    plt.clf()
    fig, axes = plt.subplots(rows, cols, figsize = (9*cols, 3*rows))
    axes = axes.ravel()

    for i in range(state_dim):
        ax = axes[i]
        X = range(len(model_states[i]))

        ax.fill_between(X, means[i]+stds[i], means[i]-stds[i], color='r', alpha=0.5)
        ax.plot(env_states[i],   color='k')
        ax.plot(model_states[i], color='b')
        ax.plot(means[i], color='r')

        if i == 0:
            ax.set_title('reward')
        elif i == 1:
            ax.set_title('terminal')
        else:
            ax.set_title('state dim {}'.format(i-2))

    # Plot conf score
    ax = axes[-1]
    ax.plot(conf_r, color='k', marker='x')
    ax.plot(conf_f, color='b', marker='x')
    ax.set_title('confidence score')

    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', layout = 'tight')
    buf.seek(0)

    img = cv2.imdecode(np.fromstring(buf.getvalue(), dtype=np.uint8), -1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.transpose(2,0,1) / 255.
    
    writer.add_image(label, img, epoch)

    plt.close()


'''
    writer video : [ batch x channels x timesteps x height x width ]
'''
def record_trajectories(writer, label, epoch, env_images, model_images=None):
    traj_length = len(env_images)
    if model_images is not None:
        assert len(env_images) == len(model_images)
        images = [np.concatenate((env_img, model_img)) for (env_img, model_img) in zip(env_images, model_images)]
    else:
        images = env_images
        
    ## [ traj_length, 2 * H, W, C ]
    images = np.array(images)
    images = torch.Tensor(images)

    ## [ traj_length, C, 2 * H, W ]
    images = images.permute(0,3,1,2)
    ## [ B, traj_length, C, 2 * H, W ]
    images = images.unsqueeze(0)

    images = images / 255.
    images = images[:,:,0].unsqueeze(2)

    print('[ Visualization ] Saving to {}'.format(label))
    fps = min(max(traj_length / 5, 2), 30)
    writer.add_video('video_' + label, images, epoch, fps = fps)


def visualize_policy(real_env, fake_env, policy, disc, writer, timestep, max_steps=100, focus=None, label='model_vis', img_dim=128):
    init_obs = real_env.reset()
    obs = init_obs.copy()

    observations_r = [obs]
    observations_f = [obs]
    rewards_r = [0]
    rewards_f = [0]
    terminals_r = [False]
    terminals_f = [False]
    means_f = [np.concatenate((np.zeros(2), obs))]
    stds_f = [np.concatenate((np.zeros(2), obs*0))]
    actions = []
    conf_r = []
    conf_f = []

    i = 0
    term_r, term_f = False, False
    while not (term_r and term_f) and i <= max_steps:

        act = policy.actions_np(obs[None])[0]
        if not term_r:
            next_obs_r, rew_r, term_r, info_r = real_env.step(act)
            observations_r.append(next_obs_r)
            rewards_r.append(rew_r)
            terminals_r.append(term_r)

            # add discriminator score
            score = disc.predict(np.concatenate([obs, next_obs_r, [rew_r]]))
            conf_r.append(score)

        if not term_f:
            next_obs_f, rew_f, term_f, info_f = fake_env.step(obs, act)
            observations_f.append(next_obs_f)
            rewards_f.append(rew_f)
            terminals_f.append(term_f)
            means_f.append(info_f['mean'])
            stds_f.append(info_f['std'])

            # add discriminator score
            score = disc.predict(np.concatenate([obs, next_obs_f, rew_f]))
            conf_f.append(score)
        
        actions.append(act)

        if not term_f:
            obs = next_obs_f
        else:
            obs = next_obs_r

        i += 1

    terminals_r = np.array([terminals_r]).astype(np.uint8).T
    terminals_f = np.array([terminals_f]).astype(np.uint8).T
    rewards_r = np.array([rewards_r]).T
    rewards_f = np.array([rewards_f]).T

    rewards_observations_r = np.concatenate((rewards_r, terminals_r, np.array(observations_r)), -1)
    rewards_observations_f = np.concatenate((rewards_f, terminals_f, np.array(observations_f)), -1)
    plot_trajectories(writer, label, timestep, rewards_observations_r, rewards_observations_f, means_f, stds_f, conf_r, conf_f)
    #record_trajectories(writer, label, epoch, images_r)

def visualize_model_perf(real_env, fake_env, policy, writer, timestep, max_steps=1000, focus=None, label='model_perf', img_dim=128):
    init_obs = real_env.reset()
    obs = init_obs.copy() #(17, ) for walker

    plotter = Plotter(obs.shape[0])

    step = 0
    while step < max_steps:
        act = policy.actions_np(obs[None])[0]

        next_obs_r, rew_r, term_r, info_r = real_env.step(act)
        next_obs_f, rew_f, term_f, info_f = fake_env.step(obs, act)

        plotter.add_next_state_dp(0, 
                torch.from_numpy(next_obs_r).view(1, -1).float(),
                torch.from_numpy(next_obs_f).view(1, -1).float())
        plotter.add_reward_dp(0,
                torch.tensor(rew_r).view(1, -1).float(),
                torch.tensor(rew_f).view(1, -1).float())

        if term_r:
            init_obs = real_env.reset()
            obs = init_obs.copy()
        else:
            obs = next_obs_r

        step += 1

    # plotter.dump_plots('/home/timchu/mbpo-torch', [0])
    plotter.dump_plots(None, [0], writer, label, timestep)

