import os
import sys
from DQN import Model
import traci
from SimulationEnv import SumoEnv
import numpy as np
import random
from Traffic_generator_weibu import TrafficGenerator
from collections import deque
import dataPlotter
from keras.models import load_model
import copy

class TLAgent:
    
    def __init__(self, env, traffic_gen, max_steps, num_experiments, total_episodes, qmodel_filename, stats, init_epoch, learn=True):
        """
        Constructor for the TLAgent class.

        Args:
            env (SumoEnv): Sumo environment object.
            traffic_gen (TrafficGenerator): Traffic generator object.
            max_steps (int): Maximum steps per episode.
            num_experiments (int): Number of experiments to run.
            total_episodes (int): Total number of episodes per experiment.
            qmodel_filename (str): Filename for the Q-model.
            stats (dict): Statistics dictionary.
            init_epoch (int): Initial epoch.
            learn (bool, optional): Whether to train the agent. Defaults to True.
        """
        
        self.env = env
        self.traffic_gen = traffic_gen
        self.total_episodes = total_episodes
        self.discount = 0.75
        self.epsilon = 0.9
        self.replay_buffer = deque(maxlen=50000)
        self.batch_size = 100
        self.num_states = 80
        self.num_actions = 4
        self.num_experiments = num_experiments

        # Phases are in the same order as specified in the .net.xml file
        self.PHASE_NS_GREEN = 0  # action 0 code 00
        self.PHASE_NS_YELLOW = 1
        self.PHASE_NSL_GREEN = 2  # action 1 code 01
        self.PHASE_NSL_YELLOW = 3
        self.PHASE_EW_GREEN = 4  # action 2 code 10
        self.PHASE_EW_YELLOW = 5
        self.PHASE_EWL_GREEN = 6  # action 3 code 11
        self.PHASE_EWL_YELLOW = 7

        self.green_duration = 10
        self.yellow_duration = 4
        self.stats = stats
        self.init_epoch = init_epoch
        self.QModel = None
        self.tau = 20
        self.TargetQModel = None
        self.qmodel_filename = qmodel_filename
        self.stats_filename = stats_filename
        self.init_epoch = init_epoch
        self._load_models(learn)
        self.max_steps = max_steps
    def _load_models( self , learn = True) :
        
        self.QModel = Model(self.num_states, self.num_actions )
        self.TargetQModel = Model(self.num_states, self.num_actions)
        
        if  self.init_epoch !=0 or not learn:
            print('model read from file')
            qmodel_fd = open(self.qmodel_filename, 'r')
                   
            if (qmodel_fd is not None):
                
                self.QModel = load_model(qmodel_fd.name)
                self.TargetQModel = load_model(qmodel_fd.name)
            
        return   self.QModel, self.TargetQModel    

             
            
    def _preprocess_input( self, state ):
        state = np.reshape(state, [1, self.num_states])
        return state
    
    def _add_to_replay_buffer( self, curr_state, action, reward, next_state, done ):
        self.replay_buffer.append((curr_state, action, reward, next_state, done))
        
    def _sync_target_model( self ):
        self.TargetQModel.set_weights( self.QModel.get_weights()) 
        
    def _replay(self):
        x_batch, y_batch = [], []
        mini_batch = random.sample( self.replay_buffer, min(len(self.replay_buffer), self.batch_size)) 
        
        for i in range( len(mini_batch)):
            curr_state, action, reward, next_state, done = mini_batch[i]
            y_target = self.QModel.predict(curr_state) # get existing Qvalues for the current state
            y_target[0][action] = reward if done else reward + self.discount*np.max(self.TargetQModel.predict(next_state)) # modify the qvalues for the action perfomrmed to get the new target 
            x_batch.append(curr_state[0])
            y_batch.append(y_target[0])
        
        self.QModel.fit(np.array(x_batch), np.array(y_batch), batch_size=len(x_batch), verbose=0)
        
        
    def _agent_policy( self, episode, state, learn = True ):
        
        if learn:
            epsilon = 1 - episode/self.total_episodes
            choice  = np.random.random()
            if choice <= epsilon:
                action = np.random.choice(range(self.num_actions))
            else:
                action =  np.argmax(self.QModel.predict(state))
        else:
            action =  np.argmax(self.QModel.predict(state))
                
        return action
        
     # SET IN SUMO THE CORRECT YELLOW PHASE
    def _set_yellow_phase(self, old_action):
        yellow_phase = old_action * 2 + 1 # obtain the yellow phase code, based on the old action
        traci.trafficlight.setPhase("TL", yellow_phase)

    # SET IN SUMO A GREEN PHASE
    def _set_green_phase(self, action):
        if action == 0:
            traci.trafficlight.setPhase("TL", self.PHASE_NS_GREEN)
        elif action == 1:
            traci.trafficlight.setPhase("TL", self.PHASE_NSL_GREEN)
        elif action == 2:
            traci.trafficlight.setPhase("TL", self.PHASE_EW_GREEN)
        elif action == 3:
            traci.trafficlight.setPhase("TL", self.PHASE_EWL_GREEN)

    
    def evaluate_model( self, experiment, seeds ):
        
        self.traffic_gen.generate_routefile(seeds[self.init_epoch])
        curr_state = self.env.start()
        
        for e in range( self.init_epoch, self.total_episodes):
            
            done = False
            sum_intersection_queue = 0
            sum_neg_rewards = 0
            old_action = None
            while not done:
                curr_state = self._preprocess_input( curr_state )
                action = self._agent_policy( e, curr_state, learn = False)
                yellow_reward = 0
                    
                if old_action!= None and old_action != action:
                    self._set_yellow_phase(old_action)
                    yellow_reward, _ , _ = self.env.step(self.yellow_duration)
                   
                self._set_green_phase(action)
                reward, next_state, done = self.env.step(self.green_duration)
                reward += yellow_reward
                next_state = self._preprocess_input( next_state )
                curr_state = next_state
                old_action = action
                sum_intersection_queue += self.env.get_intersection_q_per_step()
                if reward < 0:
                    sum_neg_rewards += reward

            self._save_stats(experiment, e, sum_intersection_queue,sum_neg_rewards)
            print('sum_neg_rewards={}'.format(sum_neg_rewards))
            print('sum_intersection_queue={}'.format(sum_intersection_queue))
            print('Epoch {} complete'.format(e))
            if e != 0:
                os.remove('stats_{}_{}.npy'.format(experiment, e-1))
            elif experiment !=0:
                os.remove('stats_{}_{}.npy'.format(experiment-1, self.total_episodes-1))
            if e +1 < self.total_episodes:
                self.traffic_gen.generate_routefile(seeds[e+1])
            curr_state =self.env.reset()

   
        
    def train( self, experiment ):
        
        self.traffic_gen.generate_routefile(0)
        curr_state = self.env.start()
   
        for e in range( self.init_epoch, self.total_episodes):
            
            curr_state = self._preprocess_input( curr_state)
            old_action =  None
            done = False # whether  the episode has ended or not
            sum_intersection_queue = 0
            sum_neg_rewards = 0
            while not done:
                    
                action = self._agent_policy( e,curr_state)
                yellow_reward = 0
                    
                if old_action!= None and old_action != action:
                    self._set_yellow_phase(old_action)
                    yellow_reward, _ , _ = self.env.step(self.yellow_duration)
                   
                self._set_green_phase(action)
                reward, next_state, done = self.env.step(self.green_duration)
                reward += yellow_reward
                next_state = self._preprocess_input( next_state )
                self._add_to_replay_buffer( curr_state, action, reward, next_state, done )
                
                if e > 0 and e % self.tau == 0:
                    self._sync_target_model()
                self._replay()
                curr_state = next_state
                old_action = action
                sum_intersection_queue += self.env.get_intersection_q_per_step()
                if reward < 0:
                    sum_neg_rewards += reward
                    
          
            self._save_stats(experiment, e, sum_intersection_queue,sum_neg_rewards)
            self.QModel.save('qmodel_{}_{}.hd5'.format(experiment, e))
            if e != 0:
                os.remove('qmodel_{}_{}.hd5'.format(experiment, e-1))
                os.remove('stats_{}_{}.npy'.format(experiment, e-1))
            elif experiment !=0:
                os.remove('qmodel_{}_{}.hd5'.format(experiment-1, self.total_episodes-1))
                os.remove('stats_{}_{}.npy'.format(experiment-1, self.total_episodes-1))
            self.traffic_gen.generate_routefile(e+1)
            curr_state = self.env.reset()   # reset the environment before every episode
            print('Epoch {} complete'.format(e))
        
    def execute( self):
        while traci.simulation.getMinExpectedNumber() > 0:
            traci.simulationStep()
            
    def _save_stats(self, experiment, episode, sum_intersection_queue_per_episode, sum_rewards_per_episode):
        self.stats['rewards'][experiment, episode] = sum_rewards_per_episode
        self.stats['intersection_queue'][experiment, episode] = sum_intersection_queue_per_episode  
        np.save('stats_{}_{}.npy'.format(experiment, episode), self.stats)
        
      
      
        """
        Constructor for the TLAgent class.

        Args:
            env (SumoEnv): Sumo environment object.
            traffic_gen (TrafficGenerator): Traffic generator object.
            max_steps (int): Maximum steps per episode.
            num_experiments (int): Number of experiments to run.
            total_episodes (int): Total number of episodes per experiment.
            qmodel_filename (str): Filename for the Q-model.
            stats (dict): Statistics dictionary.
            init_epoch (int): Initial epoch.
            learn (bool, optional): Whether to train the agent. Defaults to True.
        """
        
        self.env = env
        self.traffic_gen = traffic_gen
        self.total_episodes = total_episodes
        self.discount = 0.75
        self.epsilon = 0.9
        self.replay_buffer = deque(maxlen=50000)
        self.batch_size = 100
        self.num_states = 80
        self.num_actions = 4
        self.num_experiments = num_experiments

        # Phases are in the same order as specified in the .net.xml file
        self.PHASE_NS_GREEN = 0  # action 0 code 00
        self.PHASE_NS_YELLOW = 1
        self.PHASE_NSL_GREEN = 2  # action 1 code 01
        self.PHASE_NSL_YELLOW = 3
        self.PHASE_EW_GREEN = 4  # action 2 code 10
        self.PHASE_EW_YELLOW = 5
        self.PHASE_EWL_GREEN = 6  # action 3 code 11
        self.PHASE_EWL_YELLOW = 7

        self.green_duration = 10
        self.yellow_duration = 4
        self.stats = stats
        self.init_epoch = init_epoch
        self.QModel = None
        self.tau = 20
        self.TargetQModel = None
        self.qmodel_filename = qmodel_filename
        self.stats_filename = stats_filename
        self.init_epoch = init_epoch
        self._load_models(learn)
        self.max_steps = max_steps
        


    # The rest of the code remains unchanged

if __name__ == "__main__":
    # --- TRAINING OPTIONS ---
    training_enabled = False
    gui = 0

    sys.path.append(os.path.join('c:', os.sep, 'Users', 'Desktop', 'Work', 'Sumo', 'tools'))
    # ----------------------

    # attributes of the agent

    # setting the cmd mode or the visual mode
    if gui == False:
        sumoBinary = 'sumo.exe'
    else:
        sumoBinary = 'sumo-gui.exe'

    
    # initializations
    max_steps = 50# seconds = 1 h 30 min each episode
    total_episodes =10
    num_experiments = 1
    learn = False
    traffic_gen = TrafficGenerator(max_steps)
    qmodel_filename, stats_filename = dataPlotter.get_file_names()
    init_experiment, init_epoch = dataPlotter.get_init_epoch(stats_filename, total_episodes)
    print('init_experiment={} init_epoch={}'.format(init_experiment, init_epoch))
    stats = dataPlotter.get_stats(stats_filename, num_experiments, total_episodes)

    for experiment in range(init_experiment, num_experiments):
        env = SumoEnv(sumoBinary, max_steps)
        tl = TLAgent(env, traffic_gen, max_steps, num_experiments, total_episodes, qmodel_filename, stats, init_epoch, learn)
        init_epoch = 0  # reset init_epoch after the first experiment
        if learn:
            tl.train(experiment)
        else:
            seeds = np.load('seed.npy')
            tl.evaluate_model(experiment, seeds)
            
        stats = copy.deepcopy(tl.stats)
        print(stats['rewards'][0:experiment+1, :])
        print(stats['intersection_queue'][0:experiment+1, :])
        dataPlotter.plot_rewards(stats['rewards'][0:experiment+1, :])
        dataPlotter.plot_intersection_queue_size(stats['intersection_queue'][0:experiment+1, :])
        dataPlotter.plot_sample(stats['intersection_queue'],'smart traffic lights', 'Cumulative Vehicle Queue Size', 'Adaptive TRLC')
        dataPlotter.plot_sample(stats['rewards'],'smart traffic lights', 'cumulative_wait_times', 'Adaptive TRLC')
        
        del env
        del tl
        print('Experiment {} complete.........'.format(experiment))
