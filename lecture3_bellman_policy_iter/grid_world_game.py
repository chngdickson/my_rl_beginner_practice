import random
import pygame
BLOCK_SIZE = 30
WHITE, RED , BLUE, BLUE2, BLACK = (255,255,255), (200,0,0), (0, 0, 255), (0,100,255) , (0,0,0)

class GridWorld():
    def __init__(self,  w=10,h=10):
        self.w = w*BLOCK_SIZE
        self.h = h*BLOCK_SIZE
        self._init_display(self.w,self.h)
    
    def loop_play_step(self):
        game_over = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        self._update_ui()
        self.clock.tick(30)
        return game_over
    
    def _update_ui(self):
        text = self.font.render("press Q for bellman expectation, W for policy improvement, E Policy Iteration", True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()
    def _init_display(self,w,h):
        pygame.init()
        self.display = pygame.display.set_mode((w,h))
        self.clock = pygame.time.Clock()
        #print(pygame.font.get_fonts())
        #print(pygame.font.match_font('arial'))
        self.font = pygame.font.Font(pygame.font.match_font('arial'), 20)
        
if __name__ == '__main__':
    my_grid_world = GridWorld(20,20)
    game_over = False
    while not game_over:
        game_over = my_grid_world.loop_play_step()