import gym

env = gym.make('Pendulum-v0')

s0 = env.reset()

a0 = [0.5]

s1, r1, done, _ = env.step(a0)

print(s0)
print(s1)
print(r1)

while 1:
    env.render()