import gym
import numpy as np
import torch
import torchvision.transforms as T
from collections import deque

'''
This class does 2 Things
1. Returns stacked of processed states (1,4,84,84) (1, stack, Width, Height)
2. Add an additional reward system if it dies
'''
class AtariEnvManager():
    def __init__(self, game_env, num_images_stacked, is_use_additional_ending_criterion):
        self.game_env = game_env
        self.env = gym.make(game_env).unwrapped
        self.env.reset()
        self.current_screen = None
        self.done = False
        self.n_actions = self.env.action_space.n
        
        # Stack K images together as 1 state
        self.num_stacked = num_images_stacked
        self.running_queue = deque(maxlen=self.num_stacked)
        self.is_additional_ending = False
        self.current_lives = None
        self.is_use_additional_ending_criterion = is_use_additional_ending_criterion
        
        #self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        
    def print_action_meanings(self):
        print(self.env.get_action_meanings())
        
    def reset(self):
        self.env.reset()
        self.current_screen = None
        self.running_queue.clear()
        self.is_additional_ending = False
        self.current_lives = None
        self.black_screen = torch.zeros_like(self.get_processed_transformed_screen_state())
        
    def close(self):
        self.env.close()
    
    def render(self,mode = 'human'):
        return self.env.render(mode)
    
    def get_state(self):
        
        if self.current_screen is None:# Just started, 4 black
            [self.running_queue.append(self.black_screen) for _ in range(self.num_stacked)]
        elif self.done or self.is_additional_ending: # Game_over, 3 norm, last black_screen
            self.running_queue.append(self.black_screen)
        else: # Normal Scenario
            self.current_screen = self.get_processed_transformed_screen_state()
            self.running_queue.append(self.current_screen)
            
        self.current_screen = self.get_processed_transformed_screen_state()
        #print('Should be [1 k H W]',torch.stack(list(self.running_queue),dim=1).squeeze(2).shape)
        return torch.stack(tuple(self.running_queue),dim=1).squeeze(2) #Check shape is (1, stack_num, Height, Width)
    
    def step(self, action):
        _, reward, self.done, lives = self.env.step(action.item())
        # lives = {'ale.lives': 5}

        if self.is_use_additional_ending_criterion:
            if self.current_lives is None:# Just started
                self.current_lives = lives['ale.lives']
            elif lives['ale.lives'] < self.current_lives:
                self.is_additional_ending = True
                self.current_lives = lives['ale.lives']
                #print('Died, reward=-1')
                reward = -1
            else:
                self.is_additional_ending = False
            
        return torch.tensor([reward])
        # TODO: 
    
    def get_processed_transformed_screen_state(self):
        # Convert rgb rendering to float, rescale, convert to tensor
        screen = self.env.render('rgb_array').transpose((2, 0, 1))
        screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
        screen = torch.from_numpy(screen)
        #screen = self.crop_screen(screen)
        # Use torchvision package to compose image transforms
        resize = T.Compose([
            T.ToPILImage()
            , T.Grayscale()
            , T.Resize((84, 84)) #BZX: the original paper's settings:(110,84), however, for simplicty we...
            , T.ToTensor()
        ])
        # add a batch dimension (BCHW)
        screen = resize(screen)
        #print('Processed_shape, [1,1,84,84]',screen.unsqueeze(0).shape)
        return screen.unsqueeze(0)  # BZX: Pay attention to the shape here. should be [1,1,84,84]
    
# atari = AtariEnvManager('Breakout-v4',4,True)
# atari.reset()
# #atari.step(action=1)
# atari.get_state()
