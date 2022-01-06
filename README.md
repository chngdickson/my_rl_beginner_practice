# Project - RL Beginner Pratice

### Description
This repo is my intepretation practice of the exercises shown in David Silver's youtube course "Introduction to RL" (https://www.youtube.com/watch?v=2pWv7GOvuf0&list=PLqYmG7hTraZDM-OYHWgPebj2MfCFzFObQ&ab_channel=DeepMind). 
It includes some algorithms DDQN, A2C, Policy based, Value based and the most fundamental dynamic programming.

### Requirement
* [python 3.7](https://www.python.org) 
* [pytorch 1.0.1](https://pytorch.org/)
* [gym 0.13.1](https://github.com/openai/gym)

### Environment

Most of the environment is simulated with basic python while some of it is simulated with openai's gym.

### Examples 
here the images of some of the projects that I have done within this directory.
* Blackjack using monte_carlo control
* Cartpole dense reward problem
* Mountain Climber sparse reward problem
* Breakout using DDQN and Experience replay

##### Blackjack
This image demonstrated monte carlo control being used to demonstrate the best state we want to be in a blackjack game.
![blackjack](pictures/monte_carlo_control.png)

##### Cartpole
Cartpole is a wonderful experiment to demonstate a dense reward problem.
Dense reward is where there is a lot of positive rewards in the reward system on a constant basis.
![dense reward](pictures/cartpole.png)

##### Mountain Climber 
While cartpole demonstrates dense reward, Mountain climber is used for states which presents a sparse reward.
Much like how dota 2 will only give a reward when a kill happened, tower being taken down or a throne being taken down to achieve victory.
![sparse reward](pictures/mountain.png)

##### Breakout 
In this, Vision algorithm was used such as DDQN paired with Experience replay .
![breakout](pictures/breakoutDDQN.png)

