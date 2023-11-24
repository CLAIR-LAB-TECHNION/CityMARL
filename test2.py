from citylearn_env.city_learn_marl import setup_environment, setup_learning, train, perform_training_iteration, evaluate
from citylearn_env.custom_rewards import SACSolarReward


def main():
    env= setup_environment()
    agent_kwargs=setup_learning()
    train_result= train(env=env, agent_kwargs= agent_kwargs, episodes=3, reward_function=SACSolarReward, random_seed=0)
    evaluate(env, train_result.get('model'))

if __name__ == "__main__":
    main()