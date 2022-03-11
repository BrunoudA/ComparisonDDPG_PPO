import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import random
# import pygame
import math

#I do not comment the self parameter because it's just an instance of the class (classic parameter)

class Crosswalk_comparison2(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}
    """
        Crosswalk_comparison2: creation of the environment
        :param dt: step time
        :param Vm: max pedestrian speed
        :param tau: reaction time
        :param lower_bounds: parameters for initial state (lower bound)
        :param upper_bounds: parameters for initial state (upper bound)
        :param simulation: pedestrian speed model
    """
    
    def __init__(self, dt, Vm, tau, lower_bounds, upper_bounds, simulation="unif"):
        super(Crosswalk_comparison2, self).__init__()
        self.seed()
        self.viewer = False
        self.window = None
        # Fixed data for intersection & condition simulation
        self.dt = dt  # interval step
        self.l_b = lower_bounds
        self.u_b = upper_bounds
        self.Vm = Vm  # max pedestrian speed
        self.tau = tau # reaction time = 1 sec
        self.t_init = 0.0
        # Pedestrian model
        self.simulation = simulation
        # Max steps
        self.max_episode_steps = 90
        # crosswalk size
        self.cross = 5.0
        # Define action space: car acceleration
        self.action_space = spaces.Box(low=np.array([-1.0]),
                                       high=np.array([1.0]),
                                       shape=(1,), dtype=np.float32)
        # Define state space:
        # respectively car acceleration (0), car speed (1), car position (2), pedestrian speed (3),
        # pedestrian position (4), safety distance (5), pedestrian presence in crosswalk (6),
        # pedestrian finish crossing (7), speed difference (8), time spending (9)
        self.observation_space = spaces.Box(low=np.array([self.l_b[0], 0.0, self.l_b[2], 0.0, self.l_b[4],
                                                          -30.0, 0.0, 0.0, -self.u_b[1], 0.0]),
                                            high=np.array([self.u_b[0], self.u_b[1], 20.0, Vm, 10.0,
                                                           30.0, 1.0, 1.0, 0.0, 90.0*dt]),
                                            shape=(10,), dtype=np.float32)
        self.acc_param = [(self.u_b[0]-self.l_b[0])/2.0, (self.u_b[0]+self.l_b[0])/2.0]
    
    """
        Setting the seed: we need with the same seed for PPO and DDPG
        :param seed: seed value
        :return: seed value
    """
    def seed(self, seed=1):
        self.np_rand, seed = seeding.np_random(seed)
        return [seed]

    """
        Compute the car acceleration 
        :param value_acc: continuous action from the actor network
        :return: rescale acceleration over the model bounds (corresponding to the vehicle)
    """
    def acceleration(self, value_acc):
        if(value_acc<-1. or value_acc>1.0):
            print('pbm value acc')
        return value_acc * self.acc_param[0] + self.acc_param[1]

    """
        Compute the state parameters for the vehicle
        :param action_acc: continuous action from the actor network
        :return: vehicle acceleration, speed and position
    """
    def new_car(self, action_acc):
        acc = self.acceleration(action_acc)
        if math.isnan(acc):
            print("Acc is Nan")
        speed = self.state[1] + self.dt * acc
        if math.isnan(speed):
            print("Speed is Nan")
        pos = (acc * math.pow(self.dt, 2.0)/2.0) + (self.state[1] * self.dt) + (self.state[2])
        if math.isnan(pos):
            print("Pos is Nan")

        return acc, speed, pos

    """
        Compute the Critical Gap of the pedestrian
        :param genre: gender of the pesdestrian
        :param age: age of the pesdestrian
        :param alpha: hyperparameter
        :param sigma: noise hyperparameter
        :return: critical gap
    """
    def CG_score(self, genre, age, alpha=0.09, sigma=0.09):
        fem, child, midage, old = 0.0369, -0.0355, -0.0221, -0.1810
        gamma = math.log10(self.cross/self.Vp)
        log_val = alpha+gamma+fem*(genre == 1)+child*(age == 0)+midage*(age == 1)+old*(age == 2)
        log_val = log_val + self.np_rand.normal(loc=0.0, scale=sigma)
        return math.pow(10, log_val)

    """
        Decision-making for the pedestrian
        :return: True=cross, False=wait
    """
    def choix_pedestrian(self):
        r = self.np_rand.uniform(low=0, high=1)
        speed_car = self.state[1]
        pedestrian_estimation = self.CG
        if (self.state[2] < 4.0) * (self.state[2] > 0.0): # car_size=4
            return False
        if (speed_car == 0.0) + (self.state[2] > 4.0):
            return True
        else:
            car_time = abs(self.state[2] / speed_car)
        if (car_time > pedestrian_estimation) + (self.state[2] > 0.0):
            return True
        return False

    """
        Compute the state parameters for the pedestrian for uniform speed model 
        :return: pedestrian position and speed 
    """
    def new_pedestrian_unif(self):
        pos_p = self.state[4] + self.Vp * self.dt
        return pos_p, self.Vp

    """
        Compute the state parameters for the pedestrian for sinusoidal speed model 
        :return: pedestrian position and speed 
    """
    def new_pedestrian_sin(self):
        t = self.state[9]+ self.dt
        speed_p = (self.A*math.sin(self.w*(t-self.t0))+self.B)
        pos_p = (self.A*(-math.cos(self.w*(t-self.t0))+math.cos(self.w*self.t_init))/self.w + self.B*(t-self.t_init))
        if(pos_p>=self.cross/2 and speed_p<=self.Vp):
            pos_p, speed_p=self.new_pedestrian_unif()
        return pos_p, speed_p

    """
        Compute the state parameters for the pedestrian with respect to the environment
        :return: pedestrian position and speed 
    """
    def new_pedestrian(self, function_step):
        pp = self.state[4] + self.Vp * self.dt
        choose = True
        # Pedestrian choice
        if self.state[4] == 0.0:
            choose = self.choix_pedestrian()
            self.t0 = self.state[9]
        # Pedestrian arrives at the crosswalk
        if (self.state[4] < 0.0) * (pp >= 0.0):
            pos_p, speed_p = 0.0, (-self.state[4] / self.dt) 
            self.time_stop = 0

        # Pedestrian in crosswalk
        elif (self.state[4] >= 0.0) * (self.state[4] < self.cross) * (self.T + self.t0 > self.state[9]+ self.dt ):
            # The pedestrain waits
            if self.time_stop != 0:
                pos_p, speed_p = self.state[4], 0.0
                self.time_stop = self.time_stop - 1
                self.t0 = self.t0 + self.dt
            # The pedestrian walks
            elif (self.np_rand.uniform(low=0, high=1) < 0.99) * choose:
                pos_p, speed_p = function_step()
                #self.T = self.T-self.dt
            # The pedestrian stops
            else:
                self.time_stop = self.np_rand.randint(low=2, high=5)
                if not choose:
                    self.time_stop = 0
                pos_p, speed_p = self.state[4], 0.0
                self.t0 = self.t0 + self.dt
        # Before or after the crosswalk
        else:
            pos_p, speed_p = self.new_pedestrian_unif()
        return pos_p, speed_p

    """
        Compute safety factor
        :param x: distance between the car and the crosswalk (negative value) 
        :param v: vehicle speed
        :return: safety factor
    """
    def delta_l(self, x, v):
        return -x - (v * v / (-2.0 * self.l_b[0]) + self.tau * v)

    """
        Compute reward
        :param acc: vehicle acceleration
        :param pos: vehicle position
        :param speed: vehicle speed
        :param pos_p: pedestrian position
        :param speed_p: pedestrian speed
        :return: immediate reward with respect to the new state
    """
    def new_reward_meng(self, acc, pos, speed, pos_p, speed_p):

        dl = self.delta_l(pos, speed)
        # Safety_reward
        rew1 = 3.0 * dl * (pos <= 0.)* (dl < 0.) + 0.5 * (dl >= 0.) * (pos <= 0.)
        rew1 = rew1 * (pos_p <= self.cross) * (self.choix_voiture >= 0)
        rew1= rew1 - 30. * self.accident - 5. * self.no_safe 
        if (dl < 0.) * (pos_p <= self.cross) * (self.choix_voiture >= 0):
            self.no_safe = True
        if math.isnan(rew1):
            print("Rew1 is Nan : dl="+str(dl)+", pos="+str(pos)+", et pos_p="+str(pos_p))

        # Speed_reward
        rew2 = -2. * max(speed - self.Vc*1.1, 0.) + 1. * min(speed, 0.) #prevent extreme values
        rew2 = rew2 + 0.2 * (abs(self.state[1] - speed) < 0.5) + 1.0 * (abs(self.Vc - speed) < 0.5) #encourage low changements
        rew2 = rew2 - 2. * ((speed - self.Vc) ** 2/ self.Vc**2) - 5. * ((speed_p - self.Vp)**2 / self.Vp**2)#1.0
        if math.isnan(rew2):
            print("Rew2 is Nan : speed="+str(speed)+", et Vc="+str(self.Vc))

        # Acceleration reward
        rew3 = -0.8 * ((self.state[0] - acc)**2/(self.l_b[0])**2) + 0.5 * (abs(self.state[0] - acc) < 0.2) #0.5#/(self.l_b[0])**2)
        if math.isnan(rew3):
            print("Rew3 is Nan : prev_acc="+str(self.state[0])+", et acc="+str(acc))

        if (not self.accident) * (0. < pos_p) * (0. < pos) * (self.state[2] <= 4.) * (self.state[4] < self.cross):
            self.accident = True
            self.no_safe = True
            print("Accident!")

        # Others rewards
        rew4 = - 5. * (pos_p >= self.cross) * (pos < 4.) * (self.choix_voiture > 0) #encourage pass the crossing after the pedestrian crossing
        rew4 = rew4 - 2. * (abs(self.Vc - speed) > 0.5) *(pos > 4.0) #encourage increase speed
        rew4 = rew4 - 0.1 * (pos < 4.) * (self.choix_voiture < 0) #encourage pass the crossing
        rew4 = rew4 - 5. *(self.state[2]-pos) * (self.state[2]>pos) #the car go backward
        if math.isnan(rew4):
            print("Rew4 is Nan : accident="+str(self.accident))
        return rew1+rew2+rew3+rew4

    """
        Compute the new step: with respect to the previous state and the taken action
        :param action_ar: action
        :return: new state, immediate reward, boolean (is the new state terminal?)
    """
    def step(self, action_ar):
        
        action_acc = action_ar[0]        
        acc, speed, pos = self.new_car(action_acc) 
        if self.simulation == "sin":
            pos_p, speed_p = self.new_pedestrian(self.new_pedestrian_sin)
        else:
            pos_p, speed_p = self.new_pedestrian(self.new_pedestrian_unif)
        diff_v = speed - self.Vc
        dl = self.delta_l(pos, speed)
        t = self.state[9] + self.dt
        ped_left = (pos_p >= self.cross)
        ped_CZ = (pos_p >= 0) * (pos_p <= self.cross)
        if pos > 0.0:
            self.temps = self.temps-self.dt
        reward = self.new_reward_meng(acc, pos, speed, pos_p, speed_p)
        self.state = np.array([acc, speed, pos, speed_p, pos_p, dl,
                               ped_CZ, ped_left, diff_v, t], dtype=np.float32)
        done = (pos >= 0.0) * (self.temps <= 0.0) * (pos_p > self.cross) + (self.state[9] > 90.0*self.dt)

        return self.state, reward, bool(done), {}

    """
        Initialize a new episode
        :return: initial state of the episode
    """
    def reset(self):
        # Initial car speed
        self.Vc = self.np_rand.uniform(low=self.l_b[1], high=self.u_b[1])
        # Initial pedestrian speed
        self.Vp = self.np_rand.uniform(low=self.l_b[3], high=self.u_b[3])
        # Initial pedestrian position
        state_pos_p = self.np_rand.uniform(self.l_b[4], high=self.u_b[4])
        # Initial car position
        state_pos = self.np_rand.uniform(self.l_b[2], high=self.u_b[2])
        # Car choice
        self.choix_voiture = (self.delta_l(state_pos, self.Vc) >= 0) - (self.delta_l(state_pos, self.Vc) < 0)
        # State
        self.state = np.array([0.0, self.Vc, state_pos, self.Vp, state_pos_p, self.delta_l(state_pos, self.Vc),
                               0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        # Phare
        self.phare = 0
        # Accident and Safe state
        self.accident = False
        self.no_safe = False
        # Time after pedestrian crossing (let the vehicle accelerate)
        self.temps = self.Vc/self.u_b[0]
        
        # Pedestrian parameters
        self.time_stop = 0
        self.t0 = 0.0
        self.CG = self.CG_score(0, 2)
        self.T = self.cross / self.Vp
        if self.simulation == "sin":
            check = ((self.Vp * math.pi) / 2.0 <= self.Vm)
            self.A = check * math.pi * self.Vp / 2.0 + (not check) * (self.Vm - self.Vp) / (1.0 - (2.0 / math.pi))
            self.B = (not check)*(self.Vm - self.A)
            self.w = math.pi / self.T
        return self.state  # reward, done, info can't be included

    """
        Quick render
        :param mode: only human render using pygame
    """
    def render(self, mode='human'):
        # if self.window is None:
        #   pygame.init()
        #   self.window = pygame.display.set_mode((640, 480))
        # Discard events:
        # for event in pygame.event.get():
        #   pass
        # Positions:
        # half_width = 5.0
        # half_height = 5.0
        # x_ref = 640 / 2.0 + half_width
        # y_ref = 480 / 2.0 + half_height - 25.0
        # state_pos_p = self.state[3]
        # state_pos = self.state[2]
        # self.window.fill((0, 0, 0))
        # pygame.draw.rect(self.window, pygame.Color((255, 0, 0)),
        #                 pygame.Rect(state_pos + x_ref - half_width, 0.0 + y_ref - half_height, 2.0 * half_width,
        #                             2.0 * half_height), 2)
        # pygame.draw.rect(self.window, pygame.Color((0, 255, 0)),
        #                 pygame.Rect(0.0 + x_ref - half_width, state_pos_p + y_ref - half_height, 2.0 * half_width,
        #                             2.0 * half_height), 2)
        # pygame.display.flip()
        return None
    
    """
        Close the render
    """
    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
        # if self.window is not None:
        #    pygame.quit()
