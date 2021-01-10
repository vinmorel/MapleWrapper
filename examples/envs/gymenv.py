import gym
import time
from gym import spaces
from maplewrapper import wrapper
import numpy as np
import pydirectinput

class MapleEnv(gym.Env):
    """
    Description:
        Gym environment of MapleStory v.90 and below using extracted information from maplewrapper.
        See https://github.com/vinmorel/MapleWrapper
    Observation:
        Type: Box(26)
        Num     Observation               Min                     Max
        0       Level                     0                       Inf
        1       HP                        0                       Inf
        2       MP                        0                       Inf
        3       Experience                0                       Inf
        4       Bounding X                0                       727
        5       Bounding Y                0                       269
        6       Player X1                 0                       727
        7       Player Y1                 0                       269
        8       Mob X1 (1)                0                       727
        9       Mob Y1 (1)                0                       269
        .       Mob X1 (...)              0                       727
        .       Mob Y1 (...)              0                       269
        24      Mob X1 (10)               0                       727
        25      Mob Y1 (10)               0                       269
    Actions:
        Type: Discrete(5)
        Num   Action
        0     Walk left
        1     Walk right
        2     Attack 1
        3     Attack 2
        4     Jump
    Reward:
        Reward is the sum of gained exp minus health damage, mp consumption and time penalities
    Starting State:
        All observations are intialized according to game information
    Episode Termination:
        Episode terminates when character levels up 
    """
    
    metadata = {'render.modes': ['human']}

    def __init__(self, wrapper):
        pydirectinput.PAUSE = 0.0
        self.w = wrapper

        self.lvl, self.max_hp, self.max_mp, self.max_exp = self.w.get_basestats()

        self.Inf = np.finfo(np.float32).max
        self.B_X = 727 # Bounding box max X 
        self.B_Y = 269 # Bounding box max Y

        self.Min = np.array([0] * 26,dtype=np.float32)
        self.Max = np.array([self.Inf] * 4 + [self.B_X,self.B_Y] * 11,dtype=np.float32)

        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(self.Min, self.Max, dtype=np.float32)

        self.state = None
        self.done = None

        self.steps_beyond_done = None

        self.actions_d = {
            '0' : 'left',
            '1' : 'right',
            '2' : 'ctrl',
            '3' : 'shift',
            '4' : 'alt',
            'hp' : 'insert',
            'mp' : 'delete'
        }

    def step(self,action):
        """
        The agent takes a step in the environment.
        Parameters
        ----------
        action : int
        Returns
        -------
        ob, reward, episode_over, info : tuple
            ob : List[int]
                an environment-specific object representing your observation of
                the environment.
            reward : float
                amount of reward achieved by the previous action. The scale
                varies between environments, but the goal is always to increase
                your total reward.
            episode_over : bool
                whether it's time to reset the environment again. Most (but not
                all) tasks are divided up into well-defined episodes, and done
                being True indicates the episode has terminated. (For example,
                perhaps the pole tipped too far, or you lost your last life.)
            info : Dict
                 diagnostic information useful for debugging. It can sometimes
                 be useful for learning (for example, it might contain the raw
                 probabilities behind the environment's last state change).
                 However, official evaluations of your agent are not allowed to
                 use this for learning.
        """ 
        self.take_action(action)
        self.new_state = self.get_state()
        self.get_reward(self.state,self.new_state)
        self.state = self.new_state

        if self.state[1]/self.max_hp < 0.5:
            self.take_action('hp')

        if self.state[2]/self.max_mp <0.5:
            self.take_action('mp')

        return self.state, self.reward, self.done, {}

    def reset(self):
        self.state = self.get_state()
        self.done = 0
        return self.state

    def get_state(self):
        self.player, self.stats, self.mobs = self.w.observe()
        self.player = self.player[2:4]
        self.mobs = self.sort_mobs(self.mobs)
        state = np.concatenate((self.stats,self.player, self.mobs))
        return state

    def sort_mobs(self,mob_coords):
        if len(mob_coords) == 0:
            mobs_X1_Y1 = np.zeros(20)
        else:
            sorted_mob_coords = mob_coords[mob_coords[:,2].argsort()]
            mobs_X1_Y1 = sorted_mob_coords[:,2:4].flatten()[:20] # flattened x1,y1 columns to new array, max 20 slots
            n_mobs = len(mobs_X1_Y1)            

            if n_mobs < 20:
                buffer = 20 - n_mobs  
                mobs_X1_Y1 = np.concatenate((mobs_X1_Y1,np.zeros(buffer)))

        return mobs_X1_Y1

    def take_action(self,action):
        if action != None:
            if 'p' in str(action):
                pydirectinput.press(self.actions_d[str(action)])
                return None
            else:
                random_t_keydown = np.random.uniform(0.1,0.9) # spoof bot detection
                pydirectinput.keyDown(self.actions_d[str(action)])
                time.sleep(random_t_keydown)
                pydirectinput.keyUp(self.actions_d[str(action)])
        

    def get_reward(self,old_state,new_state):
        self.delta = new_state[:4] - old_state[:4]
        self.d_lvl, self.d_hp, self.d_mp, self.d_exp = self.delta
        
        # Default penality 
        self.reward = -0.01
        # punish hp and mp loss (hp 8x more important than mp)
        if self.d_hp < 0 :
            self.reward += self.d_hp/self.max_hp
        if self.d_mp < 0 : 
            self.reward += self.d_mp/(self.max_mp * 8)
        # reward exp gains 
        if self.d_exp > 0 :
            self.reward += (self.d_exp/self.max_exp) * 20
        # re-extract base stats if level up
        if self.d_lvl >= 1:
            self.lvl, self.max_hp, self.max_mp, self.max_exp = self.w.get_basestats()
            self.done = 1
        return self.reward

    def render(self, mode='human',close=False):
        self.w.inspect('frame')

    def close(self):
        pass

if __name__ == "__main__":   
    with wrapper("Smashy", mobs=["Red Snail"]) as w:
        env = MapleEnv(w)
        env.reset()
        while True:
            env.step(action=np.random.randint(0,4))
            print(env.reward)