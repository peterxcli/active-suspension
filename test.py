import env

# env = env.make('Pendulum-v0')
env = env.make('real')
cnt = 200
while cnt > 0:
    s = env.metering()
    print(s[0]*10, s[1]*10)
    cnt -= 1
env.exit()