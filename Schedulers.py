from Utilities import *

class NoiseSTDScheduler:
    """Linear Decay Scheduler
    """

    def __init__(self,
                 num_steps = 100,
                 start_std: float = 0.3,
                 min_std: float = 0.001,
                 max_std: float = 0.3,
                 frozen: bool = False
                 ):
        
        self.std = start_std

        self.start_std = start_std
        self.min_std = min_std
        self.max_std = max_std
        
        self.update()   # ensure std lies within range

        self.num_steps = num_steps

        self.decay = (self.start_std - self.min_std)/self.num_steps

        self.frozen = frozen

        self.current_step = 0


    def step(self):
        
        if self.frozen:
            return
        
        self.current_step += 1
        self.std -= self.decay
        self.update()   # ensure std lies within range

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
