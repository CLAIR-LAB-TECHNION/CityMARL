# System operations
import os

# Data visualization
import matplotlib.pyplot as plt

#stable_baselines3
from stable_baselines3 import SAC


# CityLearn
from citylearn.data import DataSet
from citylearn.citylearn import CityLearnEnv
from citylearn.reward_function import RewardFunction, SolarPenaltyReward
from citylearn.wrappers import NormalizedObservationWrapper, StableBaselines3Wrapper

# Simulation setup
from citylearn_env.simulation_setup import set_schema_buildings, set_schema_simulation_period, set_active_observations
from citylearn_env.utils import CustomCallback, get_loader, SACDCallback
from citylearn_env.custom_rewards import SACCustomReward, YourCustomReward

from citylearn_env.simulation_results import plot_actions, plot_rewards, plot_simulation_summary

# type hinting
from typing import List, Mapping, Tuple

# System operations
import os

# Date and time
from datetime import datetime

# type hinting
from typing import List, Mapping, Tuple


def train_your_custom_sac(
    agent_kwargs: dict, episodes: int, reward_function: RewardFunction,
    building_count: int, day_count: int, active_observations: List[str],
    random_seed: int, reference_envs: Mapping[str, CityLearnEnv] = None,
    show_figures: bool = None
) -> dict:
    """Trains a custom soft-actor critic (SACD) agent on a custom environment.

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

    # get schema
    schema = DataSet.get_schema('citylearn_challenge_2022_phase_all')

    # select buildings
    schema, buildings = set_schema_buildings(
        schema, building_count, random_seed
    )
    print('Selected buildings:', buildings)

    # select days
    schema, simulation_start_time_step, simulation_end_time_step =\
        set_schema_simulation_period(schema, day_count, random_seed)
    print(
        f'Selected {day_count}-day period time steps:',
        (simulation_start_time_step, simulation_end_time_step)
    )

    # set active observations
    schema = set_active_observations(schema, active_observations)
    print(f'Active observations:', active_observations)

    # initialize environment
    env = CityLearnEnv(schema, central_agent=True)
    sacr_env = CityLearnEnv(schema, central_agent=True)

    # set reward function
    env.reward_function = reward_function(env=env)
    sacr_env.reward_function = SACCustomReward(sacr_env)

    # wrap environment
    env = NormalizedObservationWrapper(env)
    env = StableBaselines3Wrapper(env)

    sacr_env = NormalizedObservationWrapper(sacr_env)
    sacr_env = StableBaselines3Wrapper(sacr_env)

    # initialize agent
    RANDOM_SEED = 0

    sacr_model = SAC(policy='MlpPolicy', env=sacr_env, seed=RANDOM_SEED)

    # initialize loader
    total_timesteps = episodes*(env.time_steps - 1)
    print('Number of episodes to train:', episodes)

    # initialize SAC loader
    sac_modr_loader = get_loader(max=total_timesteps)
    print('Train SAC agent...')
    #display(sac_modr_loader)

    # train SAC agent
    sacr_callback = CustomCallback(env=sacr_env, loader=sac_modr_loader)
    sacr_model = sacr_model.learn(total_timesteps=total_timesteps,
                                  callback=sacr_callback)

    # Evaluate the trained SAC model
    observations = sacr_env.reset()
    sacr_actions_list = []

    while not sacr_env.done:
        actions, _ = sacr_model.predict(observations, deterministic=True)
        observations, _, _, _ = sacr_env.step(actions)
        sacr_actions_list.append(actions)

    fig = plot_actions(sacr_actions_list, 'SAC Actions', sacr_env)
    plt.show()
    reference_envs={'SAC': sacr_env}

    # initialize SACD loader
    loader = get_loader(max=total_timesteps)
    print('Train SACD agent...')
    #display(loader)




    return {
        'random_seed': random_seed,
        'env': env,
        'model': sacr_model,
        'actions': [],
        'rewards': [],
        'agent_kwargs': agent_kwargs,
        'episodes': episodes,
        'reward_function': reward_function,
        'buildings': buildings,
        'simulation_start_time_step': simulation_start_time_step,
        'simulation_end_time_step': simulation_end_time_step,
        'active_observations': active_observations,
        'train_start_timestamp': 0,
        'train_end_timestamp': 0,
    }


def main():
    # set all plotted figures without margins
    plt.rcParams['axes.xmargin'] = 0
    plt.rcParams['axes.ymargin'] = 0

    DATASET_NAME = 'citylearn_challenge_2022_phase_all'
    schema = DataSet.get_schema(DATASET_NAME)
    root_directory = schema['root_directory']

    # change the suffix number in the next code line to a
    # number between 1 and 17 to preview other buildings
    building_name = 'Building_1'

    filename = schema['buildings'][building_name]['energy_simulation']
    filepath = os.path.join(root_directory, filename)

    RANDOM_SEED = 0

    # edit next code line to change number of buildings in simulation
    BUILDING_COUNT = 2

    # edit next code line to change number of days in simulation
    DAY_COUNT = 7

    # edit next code line to change active observations in simulation
    # NOTE: More active observations could mean longer trainer time.
    ACTIVE_OBSERVATIONS = ['hour', 'day_type']

    schema, buildings = set_schema_buildings(schema, BUILDING_COUNT, RANDOM_SEED)
    schema, simulation_start_time_step, simulation_end_time_step = \
        set_schema_simulation_period(schema, DAY_COUNT, RANDOM_SEED)
    schema = set_active_observations(schema, ACTIVE_OBSERVATIONS)

    print('Selected buildings:', buildings)
    print(
        f'Selected {DAY_COUNT}-day period time steps:',
        (simulation_start_time_step, simulation_end_time_step)
    )
    print(f'Active observations:', ACTIVE_OBSERVATIONS)

    # -------------------- CUSTOMIZE ENVIRONMENT --------------------
    # Include other observations if needed.
    # NOTE: More active observations could mean longer trainer time.
    your_active_observations = [
        'hour',
        # 'day_type'
    ]

    # ------------------ SET AGENT HYPERPARAMETERS ------------------
    your_agent_kwargs = {
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

    # --------------- SET NUMBER OF TRAINING EPISODES ---------------
    your_episodes = 30

    your_results = train_your_custom_sac(
        agent_kwargs=your_agent_kwargs,
        episodes=your_episodes,
        reward_function=YourCustomReward,
        building_count=BUILDING_COUNT,
        day_count=DAY_COUNT,
        active_observations=your_active_observations,
        random_seed=RANDOM_SEED,
        show_figures=True,
    )


if __name__ == "__main__":
    main()