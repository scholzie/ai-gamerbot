import gymnasium as gym


def main():
    # Create the CartPole environment
    env = gym.make('CartPole-v1', render_mode="human")
    env.reset()

    # Sample interaction with the environment
    for _ in range(1000):
        env.render()
        observation, reward, terminated, truncated, info = env.step(
            env.action_space.sample())
        if terminated or truncated:
            env.reset()

    env.close()


if __name__ == "__main__":
    main()
