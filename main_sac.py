from collections import deque
from typing import List

import gym
import numpy as np

from agents.algorithms.sac import SACAgent as SAC

game_config = {
    "game_path": "InvertedPendulumBulletEnv-v0",
    "unity_file": None,
    "agent": "sac"
}

agent_config = {
    "LR_ACTOR": 1e-4,
    "LR_CRITIC": 1e-4,
    "CRITIC_COUNT": 2,
    "WEIGHT_DECAY": 1e-6,
    "UPDATE_EVERY": 2,
    "POLICY_DELAY": 2,
    "BUFFER_SIZE": 1e6,
    "BATCH_SIZE": 512,
    "GAMMA": 0.99,
    "TAU": 1e-3,
    "ACTION_MIN": -1.0,
    "ACTION_MAX": 1.0,
    "SCALE": 2.0,
    "CLIP_ACTOR_GRADIENTS": False,
    "CLIP_CRITIC_GRADIENTS": False,
    "MEMORY_MODULE": "NaivePrioritizedBuffer"
}


def solve(agent: SAC, env: gym.Env, n_episodes: int = 1000, max_t: int = 1000, print_every: int = 100) -> List[float]:
    """
    :param agent : Soft Actor Critic Agent
    :param env : Gym environment
    :param n_episodes : maximum number of training episodes
    :param max_t : maximum number of time steps per episode
    :param print_every: frequency of printing information throughout iteration
    """

    scores = []
    scores_deque = deque(maxlen=print_every)

    for i_episode in range(1, n_episodes + 1):
        state = env.reset()
        agent.reset()

        score = 0

        for t in range(max_t):
            # Select an action
            action = agent.act(state)

            next_state, reward, done, info = env.step(action)

            # Take step with agent (including learning)
            agent.step(state, action, reward, next_state, done)

            # Update the score
            score += np.array(reward)

            # Roll over the state to next time step
            state = next_state

            # Exit loop if episode finished
            if np.any(done):
                break

        # Save most recent score
        scores_deque.append(score)

        # Save most recent score
        scores.append(score)

        print(
            '\rEpisode {}\tAverage Score: {:.2f}\tScore: {:.2f}'.format(i_episode, np.mean(scores_deque), score),
            end="")

        if i_episode % print_every == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
            agent.save()

        if np.mean(scores_deque) >= 30.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode,
                                                                                         np.mean(scores_deque)))
            agent.save()

    return scores


def main():
    env = gym.make(game_config["game_path"])
    env_action_space: gym.spaces.box.Box = env.action_space

    agent = SAC(state_size=env.observation_space.shape[0],
                action_size=env.action_space.shape[0],
                random_seed=1337,
                max_action=env_action_space.high,
                config=agent_config)
    solve(agent, env)


if __name__ == '__main__':
    main()
