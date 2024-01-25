from citylearn_env.city_learn_marl import setup_environment, setup_learning, learn, evaluate
from citylearn_env.custom_rewards import SACSolarReward
from citylearn_env.utils import CustomCallback
from stable_baselines3 import SAC
from citylearn_env.jupyter_utils import in_notebook
def main():
    env = setup_environment(dataset_name='citylearn_challenge_2022_phase_all', reward_func_name='SACSolarReward',
                            isCentralAgent=True, building_names=['Building_1','Building_2','Building_3','Building_4'], day_count=170, random_seed=0,
                            active_observations=['month', 'hour', 'day_type','non_shiftable_load', 'solar_generation'], active_actions=['electrical_storage'])#,'dhw_storage','heating_storage','cooling_storage'] )

    RANDOM_SEED = 0
    callback_method = CustomCallback(env=env)
    episode_count = 3

    # SAC
    learning_algorithm = setup_learning(env=env, rl_algorithm_name='SAC', random_seed=RANDOM_SEED)
    train_result= learn(env=env, learning_algorithm=learning_algorithm, callback_method=callback_method, episode_count=episode_count)
    evaluate(env, 'env1', train_result.get('model'), evaluation_name='SAC')
    env.reset()

    # SACD
    learning_algorithm = setup_learning(env=env, rl_algorithm_name='SACD', random_seed=RANDOM_SEED)
    train_result = learn(env=env, learning_algorithm=learning_algorithm, callback_method=callback_method, episode_count=episode_count)
    evaluate(env, 'env1', train_result.get('model'), evaluation_name='SACD')
    env.reset()


    # PPO
    learning_algorithm = setup_learning(env=env, rl_algorithm_name='PPO', random_seed=RANDOM_SEED)
    train_result= learn(env=env, learning_algorithm=learning_algorithm, callback_method=callback_method, episode_count=episode_count)
    evaluate(env, 'env1', train_result.get('model'), evaluation_name='PPO')
    env.reset()

    # A2C
    learning_algorithm = setup_learning(env=env, rl_algorithm_name='A2C', random_seed=RANDOM_SEED)
    train_result= learn(env=env, learning_algorithm=learning_algorithm, callback_method=callback_method, episode_count=episode_count)
    evaluate(env, 'env1', train_result.get('model'), evaluation_name='A2C')
    env.reset()

    # DDPG
    learning_algorithm =setup_learning(env=env, rl_algorithm_name='DDPG', random_seed=RANDOM_SEED)
    train_result= learn(env=env, learning_algorithm=learning_algorithm, callback_method=callback_method, episode_count=episode_count)
    evaluate(env, 'env1', train_result.get('model'), evaluation_name='DDPG')
    env.reset()

    # TD3
    learning_algorithm =setup_learning(env=env, rl_algorithm_name='TD3', random_seed=RANDOM_SEED)
    train_result= learn(env=env, learning_algorithm=learning_algorithm, callback_method=callback_method, episode_count=episode_count)
    evaluate(env, 'env1', train_result.get('model'), evaluation_name='TD3')
    env.reset()


if __name__ == "__main__":
    print(in_notebook())
    main()