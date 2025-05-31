from Utilities import *
from collections import deque
from scipy.stats import linregress

# TODO: change temp_scheduler_kwargs and run simple experiments

class TemperatureScheduler:

    def __init__(self,
                 temperature_tensor: torch.Tensor,
                 start_temp: float = 1.0,
                 min_temp: float = 0.05,
                 max_temp: float = 2.0,
                 decay_type: str = 'linear',
                 e_decay_rate: float = 0.98,
                 l_num_steps: int = 150,
                 accept_threshold: float = 400,
                 slope_threshold: float = 0.01,

                 window_size: int = 20,
                 plateau_boost: float = 1.2,
                 suppress_factor = 0.8,

                 frozen = False
                 ):
        
        self.temp = start_temp
        self.temperature_tensor = temperature_tensor
        self.temperature_tensor.fill_(start_temp)

        self.start_temp = start_temp
        self.min_temp = min_temp
        self.max_temp = max_temp

        self.decay_type = decay_type
        self.e_decay_rate = e_decay_rate
        self.l_num_steps = l_num_steps

        self.accept_threshold = accept_threshold
        self.slope_threshold = slope_threshold
        self.window_size = window_size

        self.plateau_boost = plateau_boost
        self.suppress_factor = suppress_factor

        self.past_rewards = deque(maxlen=self.window_size)

        self.l_decay = (start_temp - min_temp)/l_num_steps

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
            self.l_decay = (self.temp - self.min_temp)/(self.l_num_steps - self.current_step)
            self.temp -= self.l_decay
        elif self.decay_type == 'exponential':
            self.temp *= self.e_decay_rate

        if self.accept():
            self.temp *= self.suppress_factor
            print(f"[TempScheduler] Mean window reward is over the threshold. Decreasing temp to {np.clip(self.temp, self.min_temp, self.max_temp)}.")

        elif self.detect_plateau():
            self.temp *= self.plateau_boost
            # flushes the past rewards to wait another window_size before increasing again
            self.past_rewards.clear()
            print(f"[TempScheduler] Plateau or negative slope detected. Increasing temp to {np.clip(self.temp, self.min_temp, self.max_temp)}.")

        self.update()

    def update(self):
        self.temp = np.clip(self.temp, self.min_temp, self.max_temp)
        self.temperature_tensor.fill_(self.temp)

    def freeze(self):
        self.frozen = True

    def unfreeze(self):
        self.frozen = False

    def get(self):
        return self.temperature_tensor.item()



class TemperatureScheduler_old():

    def __init__(self,
                 temperature_tensor: torch.Tensor,
                 reward_threshold: float = 450,
                 initial_temp: float = 1.0,
                 min_temp: float=0.25,
                 max_temp: float = 2.0,
                 window_size: int = 16,
                 patience: int = 4,
                 plateau_threshold: float = 0.01,
                 bump_factor: float = 1.5,
                 decay_rate: float = 0.999,
                 frozen: bool = False):

        self.temperature_tensor = temperature_tensor  # torch tensor holding temperature param (reference, change here with torch.fill mirrors to net register)
        self.reward_threshold = reward_threshold
        self.frozen = frozen
        self.timestep = 0
        self.past_rewards = []
        
        self.initial_temp = initial_temp
        self.current_temp = initial_temp
        self.max_temp = max_temp
        self.min_temp = min_temp
        
        self.window_size = window_size
        self.patience = patience
        self.plateau_threshold = plateau_threshold
        self.bump_factor = bump_factor
        self.decay_rate = decay_rate
        self.no_improve_count = 0

        self.temperature_tensor.fill_(initial_temp)

    def freeze(self):
        self.frozen = True

    def unfreeze(self):
        self.frozen = False

    def reset(self):
        self.timestep = 0
        self.past_rewards = []
        self.temperature_tensor.fill_(self.initial_temp)

    def step(self, reward):
        if (self.frozen):
            return
        
        self.timestep += 1
        self.past_rewards.append(reward)

        if len(self.past_rewards) > self.window_size:
            self.past_rewards.pop(0)
        

        self.decay_temperature()
        self.bump_if_plateau()


    def decay_temperature(self):
        # Exponential decay per step
        new_temp = self.current_temp * self.decay_rate
        self.current_temp = max(new_temp, self.min_temp)
        self.temperature_tensor.fill_(self.current_temp)

    def bump_if_plateau(self):

        if len(self.past_rewards) < self.window_size:
            return
        
        # if any in window >= threshold reset and continue
        if any([r > self.reward_threshold for r in self.past_rewards]):
            self.no_improve_count = 0
            self.current_temp *= 0.8
            self.temperature_tensor.fill_(self.current_temp) # aggressive decay if above threshold
            return
        
        start_reward = self.past_rewards[0]
        end_reward = self.past_rewards[-1]
        
        relative_improvement = (end_reward - start_reward) / (start_reward + tol)   # ratio of improvement from final recorded step to earlier in window

        if relative_improvement < self.plateau_threshold:   # no improvement in ratio
            self.no_improve_count += 1
        else:
            self.no_improve_count = 0

        if self.no_improve_count >= self.patience:          # no improvement in ratio after patience step triggers bump by bump factor
            new_temp = min(self.current_temp * self.bump_factor, self.max_temp)
            if new_temp > self.current_temp:
                print(f"Plateau detected at step {self.timestep}. Increasing temperature {self.current_temp:.3f} -> {new_temp:.3f}")
                self.current_temp = new_temp
                self.temperature_tensor.fill_(self.current_temp)
            self.no_improve_count = 0
    

    def get(self):
        return self.temperature_tensor.item()


# to pass byref for primitive types (currently unused)
class ref():
    def __init__(self, value):
        self.value = value
