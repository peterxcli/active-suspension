import gym 
import VL53L0X
# import matplotlib.pyplot as plt
# import matplotlib.animation as animation
import time
from env_obj import observation_space, action_space
import env_obj
import Jetson.GPIO as GPIO
import random
import numpy as np
import cv2

def make(envname):
    if envname == 'real':
        return real_env()
    else :
        return gym.make(envname)

class real_env(object):
    def __init__(self):
        self.observation_space = observation_space()
        self.action_space = action_space()

        GPIO.setmode(GPIO.BOARD)
        GPIO.setwarnings(False)
        self.IN1 = 21
        GPIO.setup(self.IN1, GPIO.OUT, initial=GPIO.LOW)
        self.IN2 = 22
        GPIO.setup(self.IN2, GPIO.OUT, initial=GPIO.LOW)
        self.enA = 33
        GPIO.setup(self.enA, GPIO.OUT)
        self.p = GPIO.PWM(self.enA, 100)
        self.p.start(0)

        self.tof = VL53L0X.VL53L0X(i2c_bus=1,i2c_address=0x29)
        self.tof.open()
        self.tof.start_ranging(VL53L0X.Vl53l0xAccuracyMode.BETTER)
        self.balance_pos = 147
        self.state = np.array([])
        # self.action_high_bound = 147

    def step(self, action):
        self.output(action[0])
        state = self.metering()
        reward = self.cac_reward(state, action[0])
        return state, reward

    def reset(self):
        state = self.metering()
        self.p.ChangeDutyCycle(0)
        cnt = 0
        print("start reset")
        while abs(state[2]*10) > 4 or cnt < 200:
            cnt += 1
            state = self.metering()
            if abs(state[2]) > 4:
                cnt = 0
        print("reset complete!!")
        print("start random")
        random.seed(time.time())
        consist_time = random.randint(5, 10)
        action = -99
        # if action % 2 == 0: action *= -1
        pre_time = time.time()
        while time.time()-pre_time < consist_time:
            self.output(action)

        print("random complete!!")
        self.p.ChangeDutyCycle(0)
        return self.metering()

    def cac_reward(self, state, action):
        ret = -(10*((state[1]*10)**2) + 10*((state[2]*10)**2) + (action/50.0)**2) 
        # ret = -(100*((state[1]*10)**3) + 100*abs(((state[2]*10)**3)) + (action/50.0)**3)*0.01 
        # ret = -((state[1]*10)**2 + (state[2]*10)**2 + (action/50.0)**2)
        # ret = -((state[1]*10)**2 + (state[2]*10)**2 + (action/100.0)**2)
        
        return np.array([ret]).astype(np.float)

    def output(self, voltage):
        if voltage > 0:
            GPIO.output(self.IN1, GPIO.HIGH)
            GPIO.output(self.IN2, GPIO.LOW)
            self.p.ChangeDutyCycle(max(abs(voltage), 0))
        else :
            GPIO.output(self.IN1, GPIO.LOW)
            GPIO.output(self.IN2, GPIO.HIGH)
            self.p.ChangeDutyCycle(max(abs(voltage), 0))

        # self.p.ChangeDutyCycle(abs(voltage))
        

    def metering(self):
        distance0 = self.tof.get_distance()
        distance1 = self.tof.get_distance()
        ret = np.array([distance1, abs(distance1 - distance0), distance1-self.balance_pos]).astype(np.float) #[pos, velocity, distance_between_cur_pos_and_balance_pos, ]
        ret /= 10.0
        return ret

    def exit(self):
        self.p.stop()
        GPIO.cleanup()

if __name__ == "__main__":
    env = make('real')
    print(env.observation_space.shape[0])
    print(env.action_space.shape[0])
    s0 = env.reset()
    ep = 0

    while ep < 200:
        action = np.array([random.randint(0, 100)]).astype(np.float32)
        s1, r1 = env.step(action)
        print("EP:", ep, ":", action, s1, r1)
        s0 = s1
        ep += 1

    env.p.stop()
    GPIO.cleanup()
