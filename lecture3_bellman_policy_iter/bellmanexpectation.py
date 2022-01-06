import numpy as np
WORLD_SIZE = 4

#xy = [y,x]

TERMINAL_STATES = [[0,0],[3,3]]
DISCOUNT = 1
POLICY = 0.25 # AKA ACTION probability mapping
ACTIONS_FIGS=[ '↑', '↓' , '←', '→'] 

# up, down, left, right
ACTIONS = [np.array([-1, 0]),
           np.array([1, 0]),
           np.array([0, -1]),
           np.array([0, 1])]

def step(state, action):
    # next_state = current_state + action
    next_state = (np.array(state) + action).tolist()
    next_state = if_collision(next_state,state)
    reward = -1
    return next_state, reward

def if_collision(next_state,state):
    # if next_state is outside of boundary, return it back to previous state(x,y)
    x, y = next_state
    if x < 0 or x >= WORLD_SIZE or y < 0 or y >= WORLD_SIZE:
        next_state = state

    return next_state


def bellman_expectation_equation():
    k = 0
    k_to_print = [0,1,2,3,10]
    value = np.zeros((WORLD_SIZE, WORLD_SIZE))
    while True:
    # keep iteration until convergence
        new_value = np.zeros_like(value)
        for i in range(WORLD_SIZE):
            for j in range(WORLD_SIZE):
                for action in ACTIONS:
                    (next_i, next_j), reward = step([i, j], action)
                    if [i,j] not in TERMINAL_STATES:
                        # bellman expectation equation
                        new_value[i, j] += POLICY * (reward + DISCOUNT * value[next_i, next_j])
        if k in k_to_print:
            print('k = ',k,'\n', np.round(new_value,decimals=2))
        if np.sum(np.abs(value - new_value)) < 1e-4:
            print('k = infinity',k,'\n',np.round(new_value, decimals=2))
            break
        k +=1
        value = new_value

def bellman_expectation_and_policy_improvement():
    k = 0
    k_to_print = [0,1,2,3,10]
    value = np.zeros((WORLD_SIZE,WORLD_SIZE))
    # Initialize policy at each state as 0.25
    policy = np.zeros((WORLD_SIZE,WORLD_SIZE))+0.25 
        # Bellman Expectation Eq :
        # Get V(s)pi
    while True:
        new_value = np.zeros_like(value)
        for i in range(WORLD_SIZE):
            for j in range(WORLD_SIZE):
                if [i,j] not in TERMINAL_STATES:
                    for action in ACTIONS:
                        (next_i, next_j), reward = step([i,j], action)
                        # bellman expectation equation
                        
                        new_value[i, j] += policy[i,j] * (reward + DISCOUNT * value[next_i,next_j])
        if k in k_to_print:
            print('k = ',k,'\n', np.round(new_value,decimals=2))
        if np.sum(np.abs(value - new_value)) < 1e-4:
            optimal_values = new_value
            print('k = infinity', k, '\n', np.round(new_value, decimals=2))
            break
        k += 1
        value = new_value

    # Bellman Policy Improvement:
    # Get pi/policy for each state 

    new_policy = np.empty((WORLD_SIZE,WORLD_SIZE),dtype=list)

    for i in range(WORLD_SIZE):
        for j in range(WORLD_SIZE):
            if [i,j] not in TERMINAL_STATES:
                next_values = []
                for action in ACTIONS:
                    (next_i, next_j), reward = step([i,j], action)
                    next_values.append(optimal_values[next_i,next_j])
                best_actions=np.where(next_values == np.max(next_values))[0]
                ba = []
                for action in best_actions:
                    ba.append(ACTIONS_FIGS[action])
                new_policy[i,j]= ba
    
    # print array in 2D
    for i in new_policy.tolist():
        for j in i:
            print(j, end="")
        print()      

                
                        
                        
    
# https://youtu.be/Nd1-UUMVfz4?list=PLqYmG7hTraZDM-OYHWgPebj2MfCFzFObQ&t=1250       
bellman_expectation_equation()
bellman_expectation_and_policy_improvement()
