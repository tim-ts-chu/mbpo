import gym

MBPO_ENVIRONMENT_SPECS = (
	{
        'id': 'AntTruncatedObs-v2',
        'entry_point': (f'mbpo.env.ant:AntTruncatedObsEnv'),
    },
	{
        'id': 'HumanoidTruncatedObs-v2',
        'entry_point': (f'mbpo.env.humanoid:HumanoidTruncatedObsEnv'),
    },
	{
        'id': 'Walker2dNoisyAct-v2',
        'entry_point': (f'mbpo.env.walker:Walker2dNoisyActEnv'),
    },
	{
        'id': 'AntTruncatedObsNoisyAct-v2',
        'entry_point': (f'mbpo.env.ant:AntTruncatedObsNoisyActEnv'),
    },
)

def register_mbpo_environments():
    for mbpo_environment in MBPO_ENVIRONMENT_SPECS:
        gym.register(**mbpo_environment)

    gym_ids = tuple(
        environment_spec['id']
        for environment_spec in  MBPO_ENVIRONMENT_SPECS)

    return gym_ids
