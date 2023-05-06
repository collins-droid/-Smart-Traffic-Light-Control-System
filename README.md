# -Smart-Traffic-Light-Control-System
A smart traffic light control system using DQN reinforcement learning model, trained on simulated traffic data generated by SUMO and deployed in real-world environments with cameras, OpenCV, and YOLOv5 for object detection.
# Smart Traffic Light Control System

This project aims to optimize traffic light control using a Deep Q-Network (DQN) reinforcement learning model. The model learns from simulated traffic data generated by SUMO (Simulation of Urban Mobility) and is deployed in a real-world environment using cameras, OpenCV, and YOLOv5 for object detection.

## Modules

- DQN.py: Implements the neural network model using Keras and its training methods (predict, fit, save).
- Traffic_generator_weibu.py: Generates traffic scenarios using a Weibull distribution with 1000 cars per episode.
- DataPlotter.py: Plots the performance metrics (rewards, queue size) during training using matplotlib and seaborn.
- Main module: Integrates all the other modules, manages the training process, and interacts with the SUMO simulation.

### Real-world implementation modules

- CameraModule: Captures and preprocesses camera feeds using OpenCV.
- YOLOv5Module: Performs object detection using YOLOv5 model and extracts the required parameters.
- DataProcessing: Processes the detected parameters and converts them into the input format required by qmodel.h5.
- ControlModule: Loads qmodel.h5, provides the processed input features, and executes the control actions based on the model's output.
- Main module: Integrates all the other modules and manages the end-to-end process.

## Installation

1. Install the dependencies by running `pip install -r requirements.txt`.
2. Install SUMO (Simulation of Urban Mobility) following the instructions on the official website: https://sumo.dlr.de/docs/Installing/index.html
3. Clone the repository using `git clone https://github.com/collins-droid/Smart-Traffic-Light-Control-System.git`.
4. Run the main module using `python main.py`.

## Usage

The project can be used to train a DQN model on simulated traffic data and deploy the model in a real-world environment. Camera feeds and object detection with YOLOv5 are used to provide the necessary input features for the model, which then decides the traffic light control actions.

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

