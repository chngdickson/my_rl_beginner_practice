import numpy as np

value = np.ones((6, 6))
print(value)
print('zero_like\n',np.zeros_like(value))

WORLD_SIZE=4
TERMINAL_STATES = [[0,0],[3,3]]
value = np.zeros((WORLD_SIZE, WORLD_SIZE))
for i in range(WORLD_SIZE):
    for j in range(WORLD_SIZE):
        if [i,j] == [0,0]:
            print(i,j)


value = np.zeros((WORLD_SIZE,WORLD_SIZE)) + 0.25
print(value)

while True:
    while True:
        print('loop 1')
        break
    for i in range(1,5):
        print(i)
    break

some = [1,2]
ACTIONS_FIGS=[ '←', '↑', '→', '↓'] 

for action in some:
    print(ACTIONS_FIGS[action])

new_policy = np.empty((3,3),dtype=list)
new_policy[0,0] = [0,2]
print(new_policy.tolist())

somelist = [1,2,3,4]
for i in range(3):
    print(somelist[::-1][i+1:])
    
for s in range(11):
    print(s)

import numpy as np
print(np.arange(4))