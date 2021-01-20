import gym
from MapleEnv_v4 import MapleEnv
from stable_baselines import PPO1
from maplewrapper import wrapper
from stable_baselines.common.policies import MlpPolicy


with wrapper("smashy",["Cynical Orange Mushroom"]) as w:
    env = MapleEnv(w)
    model = PPO1(MlpPolicy, env, verbose=1)
    model.learn(total_timesteps=30000, log_interval=1)
