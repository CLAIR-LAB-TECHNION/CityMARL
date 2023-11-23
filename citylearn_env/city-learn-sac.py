def train_sb_sac(
    agent_kwargs: dict, episodes: int, reward_function: RewardFunction,
    building_count: int, day_count: int, active_observations: List[str],
    random_seed: int, reference_envs: Mapping[str, CityLearnEnv] = None,
    show_figures: bool = None
) -> dict:
    """Trains a custom soft-actor critic (SAC) agent on a custom environment.

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

    model = SACD('MultiPerspectivePolicy', env, **agent_kwargs, seed=random_seed)
    sacr_model = SAC(policy='MlpPolicy', env=sacr_env, seed=RANDOM_SEED)

    # initialize loader
    total_timesteps = episodes*(env.time_steps - 1)
    print('Number of episodes to train:', episodes)

    # initialize SAC loader
    sac_modr_loader = get_loader(max=total_timesteps)
    print('Train SAC agent...')
    display(sac_modr_loader)

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
    display(loader)

    # initialize callback
    weights_vector = agent_kwargs['weights_vector']
    callback = SACDCallback(env=env, loader=loader, weights_vector=weights_vector)

    # train SACD agent
    train_start_timestamp = datetime.utcnow()
    model = model.learn(total_timesteps=total_timesteps, callback=callback)
    train_end_timestamp = datetime.utcnow()

    # evaluate SACD agent
    observations = env.reset()
    actions_list = []

    while not env.done:
        actions, _ = model.predict(observations, deterministic=True)
        observations, _, _, _ = env.step(actions)
        actions_list.append(actions)

    # get rewards
    rewards = callback.reward_history[:episodes]

    # plot summary and compare with other control results
    if show_figures is not None and show_figures:
        env_id = 'SACD'

        reference_envs = {env_id: env, **reference_envs}
        plot_simulation_summary(reference_envs)

        # plot actions
        plot_actions(actions_list, f'{env_id} Actions', env)

        # plot rewards
        _, ax = plt.subplots(1, 1, figsize=(5, 2))
        ax = plot_rewards(ax, rewards, f'{env_id} Rewards')
        plt.tight_layout()
        plt.show()

    else:
        pass

    return {
        'random_seed': random_seed,
        'env': env,
        'model': model,
        'actions': actions_list,
        'rewards': rewards,
        'agent_kwargs': agent_kwargs,
        'episodes': episodes,
        'reward_function': reward_function,
        'buildings': buildings,
        'simulation_start_time_step': simulation_start_time_step,
        'simulation_end_time_step': simulation_end_time_step,
        'active_observations': active_observations,
        'train_start_timestamp': train_start_timestamp,
        'train_end_timestamp': train_end_timestamp,
    }