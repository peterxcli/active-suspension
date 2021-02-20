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
        self.balance_pos = 133
        self.state = np.array([])

    def step(self, action):
        self.output(action[0])
        state = self.metering()
        state = np.append(state, [action])
        state[1] = state[1] - self.state[1]
        state /= 100.0
        self.state = state
        reward = self.cac_reward(state)
        return state, reward

    def reset(self):
        state = self.metering()
        cnt = 0
        print("start reset")
        while abs(state[2]) <= 4 and cnt < 200:
            cnt += 1
            state = self.metering()
            if abs(state[2]) > 4:
                cnt = 0
        print("reset complete!!")
        state = np.append(state, 0) #init action
        state[1] = 0
        state /= 100.0
        self.p.ChangeDutyCycle(0)
        self.state = state
        return state

    def cac_reward(self, state):
        ret = 0.0
        ret = -(state[1]**2 + state[2]**2 + state[3]**2) 
        
        return np.array(ret)

    def output(self, voltage):
        if voltage > 0:
            GPIO.output(self.IN1, GPIO.HIGH)
            GPIO.output(self.IN2, GPIO.LOW)
        else :
            GPIO.output(self.IN1, GPIO.LOW)
            GPIO.output(self.IN2, GPIO.HIGH)

        self.p.ChangeDutyCycle(abs(voltage))

    def metering(self):
        distance = self.tof.get_distance()
        return np.array([distance, distance, distance-self.balance_pos]).astype(np.float32) #[pos, pos_diff, distance_between_cur_pos_and_balance_pos, ]

    def exit(self):
        self.p.stop()
        GPIO.cleanup()

if __name__ == "__main__":
    sine = [
        127, 134, 142, 150, 158, 166, 173, 181, 
        188, 195, 201, 207, 213, 219, 224, 229, 234, 238, 241, 
        245, 247, 250, 251, 252, 253, 254, 253, 252, 251, 250, 
        247, 245, 241, 238, 234, 229, 224, 219, 213, 207, 201, 195, 
        188, 181, 173, 166, 158, 150, 142, 134, 127, 119, 111, 103, 
        95, 87, 80, 72, 65, 58, 52, 46, 40, 34, 29, 24, 19, 15, 12, 8, 
        6, 3, 2, 1, 0, 0, 0, 1, 2, 3, 6, 8, 12, 15, 19, 24, 29, 34, 40, 
        46, 52, 58, 65, 72, 80, 87, 95, 103, 111, 119
    ]

    env = make('real')
    print(env.observation_space.shape[0])
    print(env.action_space.shape[0])
    s0 = env.reset()
    ep = 0
    idx = 0 

    while ep < 200:
        action = (sine[idx]-127)/2
        # action = 100
        s1, r1 = env.step(action)
        print("EP:", ep, ":", action, s1, r1)
        s0 = s1
        ep += 1
        idx += 1
        idx %= len(sine)
        # time.sleep(500/1000000.0)

    env.p.stop()
    GPIO.cleanup()