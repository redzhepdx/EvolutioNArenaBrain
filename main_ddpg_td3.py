import os
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
from unityagents import UnityEnvironment

from agents.algorithms.ddpg import DDPGAgent as DDPG
from agents.algorithms.td3 import TD3Agent as T3D
from agents.multi_agent import GenericMultiObserverSingleAgent as GMOS

game_config = {
    "game_path": "Crawler_Linux",
    "unity_file": "Crawler.x86_64",
    "agent": "t3d"
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
    "CLIP_ACTOR_GRADIENTS": False,
    "CLIP_CRITIC_GRADIENTS": False,
    "MEMORY_MODULE": "NaivePrioritizedBuffer"
}


def solve(agent, env, brain_name, num_agents, n_episodes=100000, max_t=10000, print_every=100):
    """
        Params
        ======
            n_episodes (int): maximum number of training episodes
            max_t (int): maximum number of timesteps per episode
            print_every (int): frequency of printing information throughout iteration
    """

    scores = []
    scores_deque = deque(maxlen=print_every)

    for i_episode in range(1, n_episodes + 1):
        env_info = env.reset(train_mode=True)[brain_name]
        agent.reset()

        score = np.zeros(num_agents)

        # Get the current state
        state = env_info.vector_observations

        for t in range(max_t):
            # Select an action
            action = agent.act(state)

            # Send the action to the environment
            env_info = env.step(action)[brain_name]

            # Get the next state
            next_state = env_info.vector_observations

            # Get the reward
            reward = env_info.rewards

            # See if episode has finished
            done = env_info.local_done

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
        scores_deque.append(score.mean())

        # Save most recent score
        scores.append(score.mean())

        print(
            '\rEpisode {}\tAverage Score: {:.2f}\tScore: {:.2f}'.format(i_episode, np.mean(scores_deque), score.mean()),
            end="")

        if i_episode % print_every == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
            agent.save()

        if np.mean(scores_deque) >= 30.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode,
                                                                                         np.mean(scores_deque)))
            agent.save()
            break

    return scores


def visualize(scores):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(1, len(scores) + 1), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()


def main():
    # Init Env
    env = UnityEnvironment(file_name=os.path.join(game_config["game_path"], game_config["unity_file"]))

    # Get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    # Reset the environment
    env_info = env.reset(train_mode=True)[brain_name]

    # Number of agents
    num_agents = len(env_info.agents)
    print('Number of agents:', num_agents)

    # Size of each action
    action_size = brain.vector_action_space_size
    print('Size of each action:', action_size)

    # Examine the state space
    states = env_info.vector_observations
    state_size = states.shape[1]
    print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
    print('The state for the first agent looks like:', states[0])

    # Instantiate the T3D Agent
    single_agent = DDPG(state_size=state_size,
                        action_size=action_size,
                        random_seed=1337,
                        config=agent_config)

    if game_config["agent"] == "td3":
        single_agent = T3D(state_size=state_size,
                           action_size=action_size,
                           random_seed=1337,
                           config=agent_config)

    agent = GMOS(agent=single_agent,
                 num_observers=num_agents)

    # Solve the environment
    scores = solve(agent=agent,
                   env=env,
                   brain_name=brain_name,
                   num_agents=num_agents,
                   n_episodes=10000)

    # Visualize the performance graph
    visualize(scores)

    # Turn off the environment
    env.close()


if __name__ == '__main__':
    main()
