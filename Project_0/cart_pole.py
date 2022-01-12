import gym

def play_cart_pole():

    env = gym.make('CartPole-v0')
    env.reset()
    env.render()



if __name__ == '__main__':
    play_cart_pole()