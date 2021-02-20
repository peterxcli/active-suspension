import gym
import env

# env = env.make('Pendulum-v0')
env = env.make('real')

s0 = env.reset()

a0 = [50]

s1, r1 = env.step(a0)
# s1 /= 100.0
print(s0)
print(s1)
print(r1)

env.exit()

# while 1:
    # env.render()
