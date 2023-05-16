# -*- coding: utf-8 -*-
"""
Created on Mon May  1 11:29:38 2023

@author:Collins
"""


"""
Adaptive Traffic Light System based on DQN
Author: collins mtonga
Date: May, 2023

This implementation is based on:
- The original DQN algorithm by DeepMind (2013)
  Source: https://arxiv.org/abs/1312.5602
- The OpenAI Gym library
  Source: https://gym.openai.com/
- The Keras library
  Source: https://keras.io/
- The SUMO toolkit for traffic simulation
  Source: https://www.eclipse.org/sumo/
-RituPandes work adaptive traffic lights
  https://github.com/RituPande/DQL-TSC.git
"""
import traci
import numpy as np

class SumoEnv:

    def __init__(self, sumoBinary, max_steps):
        self.sumoCmd = [sumoBinary, "-c", "intersection/STLs.sumocfg", "--no-step-log", "true", "--waiting-time-memory", str(max_steps), "--log","logfile.txt"]
        self.SUMO_INT_LANE_LENGTH = 500
        self.num_states = 80  # 0-79 see _encode_env_state function for details
        self.max_steps = max_steps

        self._init()

       
         
    def _init(self):
        self.current_state = None
        self.curr_wait_time = 0
        self.steps = 0
        
    def get_state(self):
        return self.current_state
    
    def start(self):
        traci.start(self.sumoCmd)
        self.current_state = self._encode_env_state()
        return self.current_state
        
    def reset(self):
        #traci.close()
        traci.start(self.sumoCmd)
        self._init()
        self.current_state = self._encode_env_state()
        return self.current_state
    
    def step(self, num_steps=1):
        if self.steps + num_steps > self.max_steps:
            num_steps = self.max_steps - self.steps
            
        for i in range(num_steps):
            traci.simulationStep()
         
        self.steps += num_steps
        self.current_state = self._encode_env_state()
        new_wait_time  = self._get_waiting_time()
        
        # calculate reward of action taken (change in cumulative waiting time between actions)
        reward = 0.9 * self.curr_wait_time - new_wait_time
        self.curr_wait_time = new_wait_time
        
        # one episode ends when all vehicles have arrived at their destination
        is_terminal = self.steps >= self.max_steps
        return (reward, self.current_state, is_terminal)
 
    def _get_waiting_time(self):
        incoming_roads = ["E3", "E5", "E4"]
        total_waiting_time = 0
        for veh_id in traci.vehicle.getIDList():
            wait_time_car = traci.vehicle.getAccumulatedWaitingTime(veh_id)
            road_id = traci.vehicle.getRoadID(veh_id)  # get the road id where the car is located
            if road_id in incoming_roads:  # consider only the waiting times of cars in incoming roads
                total_waiting_time += wait_time_car
        return total_waiting_time
    
    def get_intersection_q_per_step(self):
        halt_E3 = traci.edge.getLastStepHaltingNumber("E3")
        halt_E4 = traci.edge.getLastStepHaltingNumber("E4")
        halt_E5 = traci.edge.getLastStepHaltingNumber("E5")
        
        intersection_queue = halt_E3 + halt_E4 + halt_E5
        return intersection_queue
        
    def _encode_env_state(self):
        state = np.zeros(self.num_states)

        for veh_id in traci.vehicle.getIDList():
            lane_pos = traci.vehicle.getLanePosition(veh_id)
            lane_id = traci.vehicle.getLaneID(veh_id)
            lane_pos = self.SUMO_INT_LANE_LENGTH - lane_pos  # inversion of lane pos, so if the car is close to TL, lane_pos = 0
            lane_group = -1  # just dummy initialization
            is_car_valid = False  # flag for not detecting cars crossing the intersection or driving away from it

            # distance in meters from the TLS -> mapping into cells
            if lane_pos < 7:
                lane_cell = 0
            elif lane_pos < 14:
                lane_cell = 1
            elif lane_pos < 21:
                lane_cell = 2
            elif lane_pos < 28:
                lane_cell = 3
            elif lane_pos < 40:
                lane_cell = 4
            elif lane_pos < 60:
                lane_cell = 5
            elif lane_pos < 100:
                lane_cell = 6
            elif lane_pos < 200:
                lane_cell = 7
            elif lane_pos < 350:
                lane_cell = 8
            elif lane_pos <= 500:
                lane_cell = 9

            # Isolate the "turn left only" from "straight" and "right" turning lanes.
            # This is because TL lights are turned on separately for these sets
            if lane_id == "r_0" or lane_id == "r_1" or lane_id == "r_3":
                lane_group = 0
            elif lane_id == "r_4":
                lane_group = 1
            elif lane_id == "r_5" or lane_id == "r_6" :
                lane_group = 2
            elif lane_id == "r_3":
                lane_group = 3
            elif lane_id == "r_4" or lane_id == "r_5" or lane_id == "r_6":
                lane_group = 4
            elif lane_id == "r_1":
                lane_group = 5
            elif lane_id == "r_3" or lane_id == "r_5" or lane_id == "r_3":
                lane_group = 6
            elif lane_id == "r_3":
                lane_group = 7

            if lane_group >= 1 and lane_group <= 7:
                veh_position = int(str(lane_group) + str(lane_cell))  # composition of the two position ID to create a number in interval 0-79
                is_car_valid = True
            elif lane_group == 0:
                veh_position = lane_cell
                is_car_valid = True

            if is_car_valid:
                state[veh_position] = 1  # write the position of the car veh_id in the state array

        return state
        
    def __del__(self):
        traci.close()
