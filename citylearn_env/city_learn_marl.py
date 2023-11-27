from citylearn_env.environment_setup import customize_environment, set_schema_simulation_period
from citylearn_env.custom_rewards import SACSolarReward, SACCustomReward
from citylearn_env.simulation_results import plot_actions, plot_building_kpis

from citylearn.citylearn import CityLearnEnv
from citylearn.wrappers import NormalizedObservationWrapper, StableBaselines3Wrapper


from stable_baselines3 import SAC


import time

def setup_environment(dataset_name='citylearn_challenge_2022_phase_all',reward_func_name ='', isCentralAgent=True, building_names=['Building_1'], day_count=7, random_seed=0, active_observations=['hour', 'day_type']):

    print("setting up environment")
    schema = customize_environment(dataset_name, building_names,day_count,random_seed,active_observations)
    # initialize environment
    env = CityLearnEnv(schema, central_agent=isCentralAgent)
    # set reward function
    if reward_func_name == 'SACSolarReward':
        env.reward_function = SACSolarReward(env=env)
    else:
        env.reward_function = SACCustomReward(env=env)
    # wrap environment
    env = NormalizedObservationWrapper(env)
    env = StableBaselines3Wrapper(env)

    # select days
    schema, simulation_start_time_step, simulation_end_time_step = set_schema_simulation_period(schema, day_count, random_seed)
    print(
        f'Selected {day_count}-day period time steps:',
        (simulation_start_time_step, simulation_end_time_step)
    )
    return env

def setup_learning(env, rl_algorithm_name, random_seed=0):
    print("setting up learning")
    learning_params_dict = {
        'learning_rate': 0.001,
        'buffer_size': 1000000,
        'learning_starts': 100,
        'batch_size': 256,
        'tau': 0.005,
        'gamma': 0.99,
        'train_freq': 1,
        'weights_vector': [1, 1],
        'policy_kwargs': {'n_reward_components': 2}
    }

    # initalized the learning algorithm
    if rl_algorithm_name == 'SAC':
        learning_algorithm = SAC(policy='MlpPolicy', env=env, seed=random_seed)
    else:
        rl_algorithm_name = ''
    return [learning_algorithm,learning_params_dict]

def learn (env, learning_algorithm, callback_method, learning_params_dict: dict, episode_count: int) -> dict:
    """Trains an agent on a custom environment.
    """
    print("starting training")

    # initialize loader
    total_timesteps = episode_count * (env.time_steps - 1)
    print('Number of episodes to train:', episode_count)

    # initialize SAC loader
    #sac_modr_loader = get_loader(max=total_timesteps)
    #print('Train agent...')
    #display(sac_modr_loader)

    # train agent
    train_start_timestamp = time.time()
    rl_model = learning_algorithm.learn(total_timesteps=total_timesteps,callback=callback_method)
    train_end_timestamp = time.time()
    print("finished training")
    return {'model': rl_model,
            'train_start_timestamp': train_start_timestamp,
            'train_end_timestamp': train_end_timestamp,}

def perform_training_iteration():
    print("performing training iteration")

def evaluate(env, model):
    print("starting evaluation")

    # Evaluate the trained SAC model
    observations = env.reset()
    actions_list = []

    while not env.done:
        actions, _ = model.predict(observations, deterministic=True)
        observations, _, _, _ = env.step(actions)
        actions_list.append(actions)

    fig = plot_actions(actions_list, 'Actions', env)
    fig.show()
    envsDict = {'env1':env}

    fig_kpis= plot_building_kpis(envs=envsDict)
    fig_kpis.show()
    a = 9


