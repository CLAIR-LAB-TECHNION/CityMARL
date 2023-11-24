from citylearn_env.city_learn_marl import customize_environment, setup_learning, train, perform_training_iteration, evaluate


def main():
    customize_environment()
    setup_learning()
    train()
    perform_training_iteration()
    evaluate()

if __name__ == "__main__":
    main()