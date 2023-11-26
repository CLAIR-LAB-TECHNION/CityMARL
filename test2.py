from citylearn_env.city_learn_marl import setup_environment, setup_learning, train, perform_training_iteration, evaluate
from citylearn_env.custom_rewards import SACSolarReward
from citylearn_env.utils import CustomCallback

def main():
    env= setup_environment()
    rl_algorithm =
    [learning_algorithm, callback, learning_params_dict]=setup_learning(env=env, rl_algorithm, loader, CustomCallback)
    train_result= train(env=env, learning_algorithm=learning_algorithm, learning_params_dict= learning_params_dict, episode_count=3, reward_function=SACSolarReward, random_seed=0)
    evaluate(env, train_result.get('model'))

if __name__ == "__main__":
    main()