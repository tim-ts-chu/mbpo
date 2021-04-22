params = {
    'type': 'MBPO',
    'universe': 'gym',
    # 'domain': 'AntTruncatedObs', ## mbpo/env/ant.py
    'domain': 'AntTruncatedObsNoisyAct', ## mbpo/env/ant.py
    'task': 'v2',

    'log_dir': '/home/timchu/data/',
    'exp_name': 'defaults',

    'kwargs': {
        'epoch_length': 1000,
        'train_every_n_steps': 1,
        'n_train_repeat': 20,
        # 'n_train_repeat': 1,
        'eval_render_mode': None,
        'eval_n_episodes': 1,
        'eval_deterministic': True,

        'discount': 0.99,
        'tau': 5e-3,
        'reward_scale': 1.0,

        'model_train_freq': 250,
        'model_retain_epochs': 1,
        'rollout_batch_size': 100e3,
        'deterministic': False,
        'num_networks': 1,
        'num_elites': 1,
        'real_ratio': 0.05,
        'target_entropy': -4,
        'max_model_t': None,
        'rollout_schedule': [20, 100, 1, 25],
    }
}
