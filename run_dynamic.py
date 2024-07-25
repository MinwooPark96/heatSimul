import gymnasium as gym
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy
from env_dynamic import TemperatureControlEnv

env = TemperatureControlEnv(
    num_heating_points = 3,
    time_step=1,
    target_temperature=200.0,
    initial_temperature=200.0,
    tolerance=10.0,
    conveyor_length=500
)

check_env(env)
model = DQN('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=5000)

mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)

print(f"Mean reward: {mean_reward} +/- {std_reward}")

model.save("temperature_control_dynamic_dqn")

model = DQN.load("temperature_control_dynamic_dqn")

# 에피소드 실행 및 시각화
obs, info = env.reset()
for _ in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    print(env.encode_action(action))
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    if terminated or truncated:
        # obs, info = env.reset()
        break