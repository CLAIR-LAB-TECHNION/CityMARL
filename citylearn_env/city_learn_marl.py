from citylearn_env.environment_setup import customize_environment, set_schema_simulation_period
from citylearn_env.custom_rewards import SACSolarReward
from citylearn_env.utils import CustomCallback,get_loader


from citylearn.citylearn import CityLearnEnv
from citylearn.wrappers import NormalizedObservationWrapper, StableBaselines3Wrapper
from citylearn.reward_function import RewardFunction
from typing import List


from stable_baselines3 import SAC


import time

def setup_environment(dataset_name='citylearn_challenge_2022_phase_all', building_names=['Building_1'], day_count=7, random_seed=0, active_observations=['hour', 'day_type']):

    print("setting up learning")
    schema = customize_environment(dataset_name, building_names,day_count,random_seed,active_observations)
    # initialize environment
    env = CityLearnEnv(schema, central_agent=True)
    # set reward function
    env.reward_function = SACSolarReward(env=env)
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

def setup_learning():
    agent_kwargs = {
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
    episode_count = 30
    print("setting up learning")

    return agent_kwargs

def train (env, agent_kwargs: dict, episodes: int, reward_function: RewardFunction,
            random_seed: int) -> dict:
    """Trains an agent on a custom environment.

       Trains an SAC agent using a custom environment and agent hyperparamter
       setup and plots the key performance indicators (KPIs), actions and
       rewards from training and evaluating the agent.

       Parameters
       ----------
       agent_kwargs: dict
           Defines the hyperparameters used to initialize the SAC agent.
       episodes: int
           Number of episodes to train the agent for.
       reward_function: RewardFunction
           A base or custom reward function class.
       building_count: int
           Number of buildings to set as active in schema.
       day_count: int
           Number of simulation days.
       active_observations: List[str]
           Names of observations to set active to be passed to control agent.
       random_seed: int
           Seed for pseudo-random number generator.
       reference_envs: Mapping[str, CityLearnEnv], default: None
           Mapping of user-defined control agent names to environments
           the agents have been used to control.
       show_figures: bool, default: False
           Indicate if summary figures should be plotted at the end of
           evaluation.

       Returns
       -------
       result: dict
           Results from training the agent as well as some input variables
           for reference including the following value keys:

               * random_seed: int
               * env: CityLearnEnv
               * model: SAC
               * actions: List[float]
               * rewards: List[float]
               * agent_kwargs: dict
               * episodes: int
               * reward_function: RewardFunction
               * buildings: List[str]
               * simulation_start_time_step: int
               * simulation_end_time_step: int
               * active_observations: List[str]
               * train_start_timestamp: datetime
               * train_end_timestamp: datetime
    """
    print("starting training")


    # initialize agent
    sac_model = SAC(policy='MlpPolicy', env=env, seed=random_seed)

    # initialize loader
    total_timesteps = episodes * (env.time_steps - 1)
    print('Number of episodes to train:', episodes)

    # initialize SAC loader
    sac_modr_loader = get_loader(max=total_timesteps)
    print('Train SAC agent...')
    #display(sac_modr_loader)

    # train SAC agent
    sac_callback = CustomCallback(env=env, loader=sac_modr_loader)
    train_start_timestamp = time.time()
    sac_model = sac_model.learn(total_timesteps=total_timesteps,callback=sac_callback)
    train_end_timestamp = time.time()
    print("finished training")
    return {'model': sac_model,
            'train_start_timestamp': train_start_timestamp,
            'train_end_timestamp': train_end_timestamp,}

def perform_training_iteration():
    print("performing training iteration")

def evaluate(env, model):
    print("starting evaluation")

    # Evaluate the trained SAC model
    observations = env.reset()
    sacr_actions_list = []

    while not env.done:
        actions, _ = model.predict(observations, deterministic=True)
        observations, _, _, _ = env.step(actions)
        sacr_actions_list.append(actions)

    #fig = plot_actions(sacr_actions_list, 'SAC Actions', sacr_env)
    #plt.show()
    #reference_envs = {'SAC': sacr_env}

    ## initialize SACD loader
    #loader = get_loader(max=total_timesteps)
    #print('Train SACD agent...')
    # display(loader)

