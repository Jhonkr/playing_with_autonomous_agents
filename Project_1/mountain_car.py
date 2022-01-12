import gym
import numpy as np


def get_discrete_state(state):
    '''
    Gets the continuous state and transforms to discrete
    :param state: state
    :return: tuple
    '''
    discrete_state = (state - env.observation_space.low) / discrete_os_win_size

    return tuple(discrete_state.astype(int))

if __name__ == '__main__':

    env = gym.make('MountainCar-v0')
    LEARNING_RATE = 0.1
    DISCOUNT = 0.95  # weight measure of how important do we find feature actions "future reward vs current reward"
    EPISODES = 25000  # epochs or iterations to the agent to learn
    SHOW_EVERY = 2000  # every x episodes show that youre alive

    discrete_os_size = [20] * len(env.observation_space.high)
    discrete_os_win_size = (env.observation_space.high - env.observation_space.low) / discrete_os_size

    q_table = np.random.uniform(low=-2, high=0, size=(discrete_os_size + [env.action_space.n]))


    for episode in range(EPISODES):

        if episode % SHOW_EVERY == 0:
            print(episode)
            render = True
        else:
            render = False

        discrete_state = get_discrete_state(env.reset())
        done = False
        while not done:
            action = np.argmax(q_table[discrete_state])
            new_state, reward, done, _ = env.step(action)
            new_discrete_state = get_discrete_state(new_state)

            if render:
                env.render()
            if not done:
                max_future_q = np.max(q_table[new_discrete_state])
                current_q = q_table[discrete_state + (action, )]

                # basically Q-learning formula
                new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT + max_future_q)
                q_table[discrete_state+(action, )] = new_q

            elif new_state[0] >= env.goal_position:  # complete the challenge
                print("VICTORY in episode: ", episode)
                q_table[discrete_state + (action,)] = 0

            discrete_state = new_discrete_state

    env.close()