# -*- coding: utf-8 -*-
"""
Created on Mon May  1 14:14:30 2023

@author: HP
"""

import numpy as np
from DQN import Model
import os
from Traffic_generator_weibu import TrafficGenerator

from dataPlotter import plot_rewards, plot_intersection_queue_size
def test_dqn():
    num_states = 10
    num_actions = 5
    
    model = Model(num_states, num_actions)
    
    dummy_state = np.random.rand(1, num_states)
    dummy_action_values = model.predict(dummy_state)
    
    assert dummy_action_values.shape == (1, num_actions), "Model prediction shape mismatch"
    
    print("DQN test passed.")
def test_traffic_generator():
    max_steps = 100
    seed = 0
    
    traffic_gen = TrafficGenerator(max_steps)
    traffic_gen.generate_routefile(seed)
    
    assert os.path.exists("intersection/trips.trips.4L.xml"), "Route file not generated"
    print("Traffic Generator test passed.")
def test_data_plotter():
    num_experiments = 5
    total_episodes = 50
    
    dummy_rewards = np.random.rand(num_experiments, total_episodes)
    dummy_queue_sizes = np.random.rand(num_experiments, total_episodes)
    
    plot_rewards(dummy_rewards, total_episodes - 1)
    plot_intersection_queue_size(dummy_queue_sizes, total_episodes - 1)
    
    print("Data Plotter test passed (check plots).")


if __name__ == "__main__":
    test_dqn()
    test_traffic_generator()
    test_data_plotter()







