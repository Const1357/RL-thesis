from Utilities import *
from collections import deque
from scipy.stats import linregress

class NoiseSTDScheduler:

    def __init__(self,
                 start_std: float = 0.3,
                 min_std: float = 0.001,
                 max_std: float = 0.3,
                 decay_type: str = 'exponential', # or 'linear'
                 e_decay_rate: float = 0.98,
                 l_num_steps: int = 150,
                 accept_threshold: float = 400,
                 slope_threshold: float = 0.01,

                 window_size: int = 20,
                 plateau_boost: float = 1.2,
                 suppress_factor = 0.8,

                 frozen = False
                 ):
        
        self.std = start_std

        self.start_std = start_std
        self.min_std = min_std
        self.max_std = max_std

        self.decay_type = decay_type
        self.e_decay_rate = e_decay_rate
        self.l_num_steps = l_num_steps

        self.accept_threshold = accept_threshold
        self.slope_threshold = slope_threshold
        self.window_size = window_size

        self.plateau_boost = plateau_boost
        self.suppress_factor = suppress_factor

        self.past_rewards = deque(maxlen=self.window_size)

        self.l_decay = (start_std - min_std)/l_num_steps

        self.frozen = frozen

        self.current_step = 0


    def detect_plateau(self):
        if len(self.past_rewards) < self.window_size:
            return False
        rewards = np.array(self.past_rewards)
        x = np.arange(len(rewards))

        regr = linregress(x, rewards)
        slope = regr.slope
        pvalue = regr.pvalue

        norm_slope = slope / (rewards.mean() + tol)

        return norm_slope < self.slope_threshold and pvalue > 0.05
        

    def accept(self):
        if len(self.past_rewards) < self.window_size:
            return False
        return np.mean(self.past_rewards) > self.accept_threshold

    def step(self, current_reward):
        self.current_step += 1
        
        if self.frozen:
            return

        self.past_rewards.append(current_reward)

        if self.decay_type == 'linear':
            self.l_decay = (self.std - self.min_std)/(self.l_num_steps - self.current_step)
            self.std -= self.l_decay
        elif self.decay_type == 'exponential':
            self.std *= self.e_decay_rate

        if self.accept():
            self.std *= self.suppress_factor
            print(f"[NoiseScheduler] Mean window reward is over the threshold. Decreasing std to {np.clip(self.std, self.min_std, self.max_std)}.")

        elif self.detect_plateau():
            self.std *= self.plateau_boost
            # flushes the past rewards to wait another window_size before increasing again
            self.past_rewards.clear()
            print(f"[NoiseScheduler] Plateau or negative slope detected. Increasing std to {np.clip(self.std, self.min_std, self.max_std)}.")

        self.update()   # ensure within range

    def update(self):
        self.std = np.clip(self.std, self.min_std, self.max_std)

    def freeze(self):
        self.frozen = True

    def unfreeze(self):
        self.frozen = False

    def get(self):
        return self.std
    
    def __call__(self):
        return self.std


# to pass byref for primitive types (currently unused)
class ref():
    def __init__(self, value):
        self.value = value
