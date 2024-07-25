import gymnasium as gym
from gymnasium import spaces
import numpy as np
from simulater import Simulater

class TemperatureControlEnv(gym.Env):
    def __init__(self, 
                 num_heating_points:int = 3,
                 time_step: int = 1,
                 target_temperature: float = 200.0,
                 initial_temperature: float = 200.0,
                 tolerance: float = 10.0,
                 conveyor_length: int = 1000,
                 newton_cooling_coefficient: float = 0.001,
                 natural_temperature: float = 25.0,
                 seed: int = 42,
                 mean_conveyor_speed: float = 2.0):
        
        super(TemperatureControlEnv, self).__init__()
        
        self.simulater = Simulater(time_step, target_temperature, initial_temperature, tolerance,
                                   conveyor_length, newton_cooling_coefficient, natural_temperature,
                                   seed, mean_conveyor_speed)
        
        self.num_heating_points = num_heating_points
        self.conveyor_length = conveyor_length
        self.max_power_level = 10

        # The action space should be encoded as a Discrete space
        self.action_space = spaces.Discrete((conveyor_length + 1) * (self.max_power_level + 1) ** num_heating_points)
        
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(4 + num_heating_points*2,), dtype=np.float32)
        self.max_patient = target_temperature + tolerance
        self.min_patient = target_temperature - tolerance
        self.reset()

    def reset(self, seed=None, options=None):
        self.current_temperature = self.simulater.initial_temperature
        self.current_position = 0
        self.time = 0
        self.heating_points = [(0, 0)] * self.num_heating_points
        return np.array([self.current_position, self.current_temperature, 0, self.num_heating_points] + [0] * self.num_heating_points * 2, dtype=np.float32), {}

    def encode_action(self, action):
        powers = [0] * self.num_heating_points
        positions = [0] * self.num_heating_points
        
        for i in range(self.num_heating_points):
            positions[i] = action % (self.conveyor_length + 1)
            action //= (self.conveyor_length + 1)
            powers[i] = action % (self.max_power_level + 1)
            action //= (self.max_power_level + 1)
        
        return [(positions[i], powers[i] / 10.0) for i in range(self.num_heating_points)]

    def step(self, action):
        heating_points = self.encode_action(action)
        self.heating_points = heating_points
        
        current_speed = self.simulater.generate_speed_of_converyor()
        self.current_position += current_speed * self.simulater.time_step
        self.time += self.simulater.time_step
        
        natural_temp = self.simulater.next_temparature_by_natural_cooling(self.current_temperature, current_speed)
        
        heating_power = 0
        
        for (position, power) in heating_points:
                up = -(self.current_position - position)**2 
                down = (2 * power)**2
                heating_power += power * np.exp(up / down)
            
        natural_temp = self.simulater.next_temparature_by_heat(natural_temp, current_speed, heating_power)
        
        self.current_temperature = natural_temp

        if self.current_position >= self.simulater.conveyor_length:
            # or self.current_temperature > self.max_patient or \
            #     self.current_temperature < self.min_patient:
            
            terminated = True
        else:
            terminated = False
        
        truncated = False  # Additional early termination conditions can be set here
        
        reward = 1 / abs(self.simulater.target_temperature - self.current_temperature)
        
        if self.current_position >= self.simulater.conveyor_length:
            reward += 10

        return np.array([self.current_position, self.current_temperature, current_speed, self.num_heating_points] + [item for sublist in self.heating_points for item in sublist], dtype=np.float32), reward, terminated, truncated, {}

    def render(self, mode='human'):
        pass
    
    def close(self):
        pass