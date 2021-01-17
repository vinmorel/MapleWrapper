import gym
import time
import numpy as np
import pydirectinput
from gym import spaces
from maplewrapper import wrapper
import cv2


class MapleEnv(gym.Env):
    """
    Description:
        Gym environment of MapleStory v.90 and below using extracted information from maplewrapper.
        See https://github.com/vinmorel/MapleWrapper
    Observation:
        Type: Dict "MapleWrapper" : box(8)
        Num     Observation                 Min                     Max
        1       Player X1                   0                       825
        2       Mob X1 (1)                  0                       825
        3       Mob X1 (2)                  0                       825
        4       Mob X1 (3)                  0                       825
        5       Mob X1 (4)                  0                       825
        6       Mob X1 (5)                  0                       825
        7       Player Facing Direction     0                       1
        8       Attacked                    0                       1

    Actions:
        Type: Discrete(4)
        Num     Action
        0       Walk left
        1       Walk right
        2       Attack 1
        3       Attack 2
    Reward:
        Reward is the sum of gained exp minus health damage, mp consumption and time penalities
    Starting State:
        All observations are intialized according to game information
    Episode Termination:
        Episode terminates every 10 minutes 
    """
    
    metadata = {'render.modes': ['human']}

    def __init__(self,w):
        pydirectinput.PAUSE = 0.0

        self.w = w
        self.lvl, self.max_hp, self.max_mp, self.max_exp = self.w.get_basestats()
        self.B_X = 850 # Bounding box max X 

        self.Min = np.array([0] * 8,dtype=np.float32)
        self.Max = np.array([self.B_X] * 6 + [1] * 2 ,dtype=np.float32)

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(self.Min, self.Max, dtype=np.float32)

        self.state = None
        self.done = None
        self.reward_threshold = 20.0
        self.trials = 200
        self.steps_counter = 0
        self.id = "MapleBot"
        self.facing = None
        self.random_t_keydown = 0.01

        self.actions_d = {
            '0' : 'left',
            '1' : 'right',
            '2' : 'ctrl',
            '3' : 'shift',
            'hp' : 'pageup',
            'mp' : 'delete',
            'pickup' : 'z'
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
        self.new_p_state, new_stats = self.get_partial_state()
        self.get_reward(self.stats,new_stats)

        # Determine if attacked
        if new_stats[1] < self.stats[1]:
            self.attacked = np.array([1])
        else :
            self.attacked = np.array([0])
        self.stats = new_stats

        # Determine facing direction 
        if action == 0:
            self.facing = np.array([action])
        if action == 1:
            self.facing = np.array([action])

        self.state = np.concatenate((self.new_p_state,self.facing,self.attacked))

        # heal if necessary 
        if new_stats[1]/self.max_hp < 0.5:
            self.take_action('hp')
        # mp if necessary
        if new_stats[2]/self.max_mp < 0.2:
            self.take_action('mp')
        # random pickup
        if np.random.binomial(1,0.3):
            self.take_action('pickup')
        # terminate episode if t 
        self.current_time = time.time()
        if int(self.current_time - self.start_time) >= 600:
            self.done = 1

        return self.state, self.reward, self.done, {}

    def reset(self):
        self.take_action(1)
        self.facing = np.array([1])
        self.attacked = np.array([0])
        self.p_state, self.stats = self.get_partial_state()
        self.state = np.concatenate((self.p_state,self.facing,self.attacked))
        self.done = 0
        self.start_time = time.time()
        return self.state

    def get_partial_state(self):
        self.player, stats, self.mobs = self.w.observe()
        self.player = np.array([self.player[2]])
        self.mobs = self.sort_mobs(self.mobs)
        state = np.concatenate((self.player, self.mobs))
        return state, stats

    def sort_mobs(self,mob_coords):
        if len(mob_coords) == 0:
            mobs_X1 = np.full(5,-1)
        else:
            sorted_mob_coords = mob_coords[mob_coords[:,2].argsort()]
            mobs_X1 = sorted_mob_coords[:,2].flatten()[:5] # flattened x1,y1 columns to new array, max 5 slots
            n_mobs = len(mobs_X1)            

            if n_mobs < 5:
                buffer = 5 - n_mobs  
                mobs_X1 = np.concatenate((mobs_X1,np.full(buffer,-1)))

        return mobs_X1

    def take_action(self,action):
        if action != None:
            if 'p' in str(action):
                pydirectinput.press(self.actions_d[str(action)])
                return None
            else:
                if action in (2,3): 
                    self.random_t_keydown = np.random.uniform(0.93,1.04) # bigger delay for attacks
                else:
                    self.random_t_keydown = np.random.uniform(0.35,0.45)
                
                pydirectinput.keyDown(self.actions_d[str(action)])
                time.sleep(self.random_t_keydown)
                pydirectinput.keyUp(self.actions_d[str(action)])
        

    def get_reward(self,old_stats,new_stats):
        old_stats = np.array(old_stats)
        new_stats = np.array(new_stats)
        self.delta = new_stats - old_stats
        self.d_lvl, self.d_hp, self.d_mp, self.d_exp = self.delta
        
        # Default penality 
        self.reward = (-1*self.random_t_keydown)/45
        # punish hp and mp loss 
        if self.d_hp < 0 :
            self.reward += self.d_hp/self.max_hp
        if self.d_mp < 0 : 
            self.reward += self.d_mp/(self.max_mp * 14)
        # reward exp gains
        if self.d_exp > 0 :
            self.reward += (self.d_exp/self.max_exp) * 200
        # re-extract base stats if level up
        if self.d_lvl >= 1:
            self.lvl, self.max_hp, self.max_mp, self.max_exp = self.w.get_basestats()
        return self.reward

    def render(self, mode='human',close=False):
        self.w.inspect('frame')


if __name__ == "__main__":   
    with wrapper("Smashy", mobs=["Cynical Orange Mushroom"]) as w:
        env = MapleEnv(w)
        env.reset()
        while True:
            env.step(action=None)
            print(env.reward)