from citylearn_env.city_learn_marl import setup_environment, setup_learning, learn, evaluate
from citylearn_env.custom_rewards import SACSolarReward
from citylearn_env.utils import CustomCallback
from stable_baselines3 import SAC
from citylearn_env.jupyter_utils import in_notebook
def main():
    env= setup_environment(reward_func_name='SACSolarReward')
    RANDOM_SEED = 0
    callback_method = CustomCallback(env=env)
    [learning_algorithm, learning_params_dict]=setup_learning(env=env, rl_algorithm_name='SAC', random_seed=RANDOM_SEED)
    train_result= learn(env=env, learning_algorithm=learning_algorithm, callback_method=callback_method, learning_params_dict= learning_params_dict, episode_count=3)
    evaluate(env, 'env1', train_result.get('model'))

if __name__ == "__main__":
    print(in_notebook())
    main()