# Simulation of Smart Traffic Light System (STLS) in SUMO

## Table of Contents
- [Introduction](#introduction)
- [Stages of Simulation Setup](#stages-of-simulation-setup)
  - [Requirements Analysis](#requirements-analysis)
  - [Software Design](#software-design)
  - [Implementation](#implementation)
  - [Testing](#testing)
  - [Documentation](#documentation)
  - [Deployment](#deployment)
  - [Maintenance](#maintenance)
- [Description of Main Modules](#description-of-main-modules)
  - [1. Deep Q-Network](#1-deep-q-network)
    - [Input Parameters](#input-parameters)
    - [Output Parameters](#output-parameters)
  - [2. Traffic Generator (Weibu)](#2-traffic-generator-weibu)
  - [3. SUMO Environment](#3-sumo-environment)
    - [State Space](#state-space)
    - [Action Space](#action-space)
    - [Reward Function](#reward-function)
    - [Terminal State](#terminal-state)
  - [4. Data Plotter](#4-data-plotter)
  - [5. Main Module](#5-main-module)

## Introduction

This README provides an overview of the simulation setup for the Smart Traffic Light System (STLS) implemented in SUMO. The system aims to optimize traffic signal timings using a Deep Q-Network (DQN) in the SUMO simulation environment.

## Stages of Simulation Setup

### Requirements Analysis

The goal of the software is to optimize traffic signal timings using a Deep Q-Network (DQN) in the SUMO simulation environment.

### Software Design

The software is designed with five main components: DQN.py, Traffic_generator_weibu.py, DataPlotter.py, SumoEnv.py, and the main module.

### Implementation

Modules and their functionalities:
- DQN.py: Implements the neural network model using Keras and its training methods.
- Traffic_generator_weibu.py: Generates traffic scenarios using a Weibull distribution.
- DataPlotter.py: Plots the performance metrics during training using matplotlib and seaborn.
- SumoEnv.py: Encapsulates the interaction with the SUMO simulation.
- Main module: Integrates all the other modules and manages the training process.

### Testing

Each module was tested independently to ensure proper functionality.

### Documentation

Documentation has been provided for each module, including their functionalities, inputs, outputs, and usage on GitHub.

### Deployment

The software is ready for use and can be distributed along with its dependencies (Python, Keras, SUMO, etc.) and installation instructions.

### Maintenance

Continuous monitoring and updates are performed to implement features, bug fixes, and performance improvements as needed.

## Description of Main Modules

### 1. Deep Q-Network

#### Input Parameters

| Description           | Explanation                                           |
|-----------------------|-------------------------------------------------------|
| Vehicle count         | Number of vehicles in each lane approaching the intersection |
| Vehicle speed         | Average speed of vehicles in each lane               |
| Traffic light state   | Current state of the traffic lights for each lane    |
| Vehicle waiting time  | Average waiting time of vehicles in each lane        |
| Lane occupancy        | Percentage of lane occupancy in each lane            |

#### Output Parameters

| Description                     | Explanation                                               |
|---------------------------------|-----------------------------------------------------------|
| Change the traffic light state | Change the traffic light state for a specific lane        |
| Extend the green light duration| Extend the green light duration for a specific lane       |
| Shorten the green light duration| Shorten the green light duration for a specific lane      |

### 2. Traffic Generator (Weibu)

This module generates traffic scenarios for the SUMO simulation using a Weibull distribution.

### 3. SUMO Environment

#### State Space

The state space consists of a discrete representation of the positions of vehicles approaching the traffic light.

#### Action Space

The action space consists of the possible traffic light configurations.

#### Reward Function

The reward function encourages the agent to minimize the waiting time of vehicles at the intersection.

#### Terminal State

A terminal state is reached when the total number of steps in the simulation reaches the maximum allowed steps.

### 4. Data Plotter

This module plots performance metrics collected during training.

### 5. Main Module

The main module integrates all other modules and oversees the simulation process.

