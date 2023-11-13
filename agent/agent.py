"""
With reference to:
https://github.com/calvinfeng/machine-learning-notebook/blob/master/reinforcement_learning/deep_q_learning.py
https://calvinfeng.gitbook.io/machine-learning-notebook/unsupervised-learning/reinforcement-learning/reinforcement_learning#deep-q-learning-example 
https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html 
https://www.youtube.com/watch?v=wc-FxNENg9U&t=217s 
"""
import Models.DQL.nnetwork as dqlnn
import numpy as np
import random
import torch
import torch.optim as optim
from utils import get_input_layer_2 as input
import Models.DQL.state as State
import game.flappyNoGraphics as Game
import game.wrapped_flappy_bird as GameVisual
from collections import deque
import pickle

class Agent(object):
    def __init__(self):
        """
        Porperties:
            gamma (float): Future reward discount rate.
            epsilon (float): Probability for choosing random policy.
            epsilon_decay (float): Rate at which epsilon decays toward zero.
            learning_rate (float): Learning rate for Adam optimizer.

        Returns:
            Agent
        """
        # constant parameters
        self.gamma = 0.95
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.996
        self.lr = 0.00005
        self.batch_size = 64
        self.max_mem_size = 50000
        # self.input_dims = 7 * 4

        #variable parameters
        self.epsilon = 0.01
        self.mem_cntr = 0
        self.mem_cntr_successful = 0

        # initializing memory
        self.memory = deque(maxlen=self.max_mem_size)
        self.memory_successful = deque(maxlen=1000)
        self.episodic_memory = []

        #initialize networks
        self.network = dqlnn.Network(self.lr)
        # self.target_net = dqlnn.Network(self.lr)
        # self.optimizer = optim.AdamW(self.network.parameters(), lr=self.lr, amsgrad=True)

    def save_experience(self):
        with open('Models/DQL/experience.pickle', 'wb') as handle:
            pickle.dump(self.memory, handle)
        with open('Models/DQL/experience_successful.pickle', 'wb') as handle:
            pickle.dump(self.memory_successful, handle)

    def load_experience(self):
        with open('Models/DQL/experience.pickle', 'rb') as handle:
            self.memory = pickle.load(handle)
        with open('Models/DQL/experience_successful.pickle', 'rb') as handle:
            self.memory_successful = pickle.load(handle)

    def getMemory(self):
        return self.memory

    def nextEpisode(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def getepsilon(self):
        return self.epsilon

    def remember(self, state, action, reward, next_state, terminal):
        if (self.mem_cntr >= self.max_mem_size - 2):
            for i in range(self.max_mem_size - 3000):
                self.memory.popleft()
            self.mem_cntr = len(self.memory) - 1

        memory = [state, action, reward, next_state, terminal]
        self.memory.append(memory)

        self.mem_cntr += 1

    def select_action(self, state):
        if np.random.rand() <= self.epsilon:
            # exploration

            # 2 in 30 = averages about 1 press every 0.5 seconds which is in the ballpark of whats required to play the game. 
            # Gives bot best start possible (as it actually has a chance of making it through the first block!)
            # in flappy bird a flap changes the gamestate a lot more than a no-flap.
            determiner = np.random.randint(0, 30);
            if (determiner <= 2):
                return 1
            return 0
        else:
            # exploitation
                state_tensor = torch.tensor([state]).to(self.network.device, dtype=torch.int32)
                action = torch.argmax(self.network.forward(state_tensor)).item()
                
        return action
    
    def updateEpsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
    def updateEpsilonScore(self, score):
        modifier = -0.01*score + 1.08
        episolon_new = min(self.epsilon * modifier, 0.7 )
        self.epsilon = max(self.epsilon_min, episolon_new)

    def learn(self):
        if self.mem_cntr < self.batch_size:
            return
        

        self.network.optimizer.zero_grad()
        max_mem = min(self.mem_cntr, self.max_mem_size)
        batch = np.random.choice(max_mem, self.batch_size, replace=False)

        batch_index = np.arange(self.batch_size, dtype=np.int32)

        # memory = [state, action, reward, next_state, game_over, score, next_reward]



        state_batch = torch.tensor([self.memory[i][0] for i in batch]).to(self.network.device, dtype=torch.float32)
        action_batch = torch.tensor([self.memory[i][1] for i in batch])
        reward_batch = torch.tensor([self.memory[i][2] for i in batch]).to(self.network.device, dtype=torch.float32)
        new_state_batch = torch.tensor([self.memory[i][3] for i in batch]).to(self.network.device, dtype=torch.float32)
        game_over_batch = torch.tensor([self.memory[i][4] for i in batch]).to(self.network.device, dtype=torch.bool)
        next_reward_batch = torch.tensor([self.memory[i][6] for i in batch]).to(self.network.device, dtype=torch.float32)

        q_current = self.network.forward(state_batch)[batch_index, action_batch]
        # q_next = self.network.forward(new_state_batch)
        # q_next[game_over_batch] = 0.0

        # max returns value as well as index, we only require index
        # q_current = reward_batch
        # q_target = next_reward_batch

        q_current = self.network.forward(state_batch)[batch_index, action_batch]
        q_next = self.network.forward(new_state_batch)

        # ask tutor how to make this part not short sighted.
        q_target = reward_batch + self.gamma * torch.max(q_next, dim=1)[0]


        loss = self.network.loss(q_target, q_current).to(self.network.device)
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.network.parameters(), 100)
        self.network.optimizer.step()

    def learn_successful(self):
        if self.mem_cntr_successful < self.batch_size:
            return
        
        # print("learning successful")
        self.network.optimizer.zero_grad()
        max_mem = min(self.mem_cntr_successful, self.max_mem_size)
        batch = np.random.choice(max_mem, self.batch_size, replace=False)

        batch_index = np.arange(self.batch_size, dtype=np.int32)

        # memory = [state, action, reward, next_state, game_over, score]



        state_batch = torch.tensor([self.memory[i][0] for i in batch]).to(self.network.device, dtype=torch.float32)
        action_batch = torch.tensor([self.memory[i][1] for i in batch])
        reward_batch = torch.tensor([self.memory[i][2] for i in batch]).to(self.network.device, dtype=torch.float32)
        new_state_batch = torch.tensor([self.memory[i][3] for i in batch]).to(self.network.device, dtype=torch.float32)
        game_over_batch = torch.tensor([self.memory[i][4] for i in batch]).to(self.network.device, dtype=torch.bool)
        next_reward_batch = torch.tensor([self.memory[i][6] for i in batch]).to(self.network.device, dtype=torch.float32)

        q_current = self.network.forward(state_batch)[batch_index, action_batch]
        # q_next = self.network.forward(new_state_batch)
        # q_next[game_over_batch] = 0.0

        # max returns value as well as index, we only require index
        # q_current = reward_batch
        # q_target = next_reward_batch
        q_current = self.network.forward(state_batch)[batch_index, action_batch]
        q_next = self.network.forward(new_state_batch)
        q_next[game_over_batch] = 0.0
        q_target = reward_batch + self.gamma * torch.max(q_next, dim=1)[0]


        loss = self.network.loss(q_target, q_current).to(self.network.device)
        loss.backward()
        self.network.optimizer.step()

    def update_episodic_memory(self, state, action, reward, next_state, done, score, current_step):
        self.episodic_memory.append([state, action, reward, next_state, done, score, 0])
        # for i in range(current_step):
        #     if (abs(current_step - i) < 10):
        #         gamma = 1.0**(current_step - i)
        #     else:
        #         gamma = self.gamma**((current_step - i)-10)
        #     self.episodic_memory[i][2] = self.episodic_memory[i][2] + (gamma) * reward
        #     # update next_reward
        # for i in range(current_step - 1):
        #     self.episodic_memory[i][6] = self.episodic_memory[i + 1][2]
        # self.episodic_memory[current_step][6] = self.episodic_memory[current_step][2]


import keyboard
import matplotlib.pyplot as plt
import Models.DQL.human_training as Trainer


def test():

    agent = Agent()
    scores, median_scores, eps_history = [], [], []
    n_games = 100000
    success_threshold = 15
    trainer = Trainer.Trainer(agent)

    # trainer.play(10)
    # agent.save_experience()
    agent.load_experience()
    for i in range(100):
        agent.learn()

    for i in range(n_games):
        game = Game.GameState()
        if (keyboard.is_pressed("p")):
            game = GameVisual.GameState()
        score = 0
        game_over = False
        state_manager = State.StateManager(4)
        state = state_manager.get()
        done = False
        # state, action, reward, next_state, done, score
        agent.episodic_memory = []
        current_step = 0
        while not done:
            action = agent.select_action(state)
            _, reward, _ = game.frame_step(action)
            state_manager.push(game)
            next_state = state_manager.get()
            if (reward == -5):
                done = True
                final_score = score
                reward = -5
            score += reward
            # agent.remember(state, action, reward, next_state, done, score)
            agent.update_episodic_memory(state, action, reward, next_state, done, score, current_step)
            

            state = next_state
            current_step += 1
        agent.learn()
        agent.learn_successful()

        agent.updateEpsilon()

        eps_history.append(agent.epsilon)
        for frame in agent.episodic_memory:
            agent.remember(frame[0], frame[1], frame[2], frame[3], frame[4], frame[5], frame[6])
        if (score > success_threshold):
            agent.remember_successful(frame[0], frame[1], frame[2], frame[3], frame[4], frame[5], frame[6])
        # agent.remember(state, action, reward, next_state, done, score)

        median_score = np.mean(scores[-100:])
        success_threshold = max(success_threshold, median_score + 10)
        # if (i % 500 == 0):
        #     success_threshold = 15
        scores.append(score)
        median_scores.append(median_score)
        # if (median_score < 20):
        #     agent.updateEpsilonScore(median_score)
        print('episode: ', i,'score: %.2f' % score,
                ' median score %.2f' % median_score, 'epsilon %.2f' % agent.epsilon)
        # if (score > 10):
        #     print("successful")
        if (keyboard.is_pressed("`")):
            break

    plt.plot(median_scores)
    plt.show()



    
