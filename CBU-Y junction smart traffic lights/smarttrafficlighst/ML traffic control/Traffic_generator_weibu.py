import numpy as np
import math

# HANDLE THE GENERATION OF VEHICLES IN ONE EPISODE
class TrafficGenerator:
    def __init__(self, max_steps):
        self._n_cars_generated = 1000  # how many cars per episode
        self._max_steps = max_steps

    # generation of routes of cars
    def generate_routefile(self, seed):
        
        if np.any(seed) >=0 :
            np.random.seed(seed)  # make tests reproducible

        # the generation of cars is distributed according to a weibull distribution
        timings = np.random.weibull(2, self._n_cars_generated)
        timings = np.sort(timings)

        # reshape the distribution to fit the interval 0:max_steps
        car_gen_steps = []
        min_old = math.floor(timings[1])
        max_old = math.ceil(timings[-1])
        min_new = 0
        max_new = self._max_steps
        for value in timings:
            car_gen_steps = np.append(car_gen_steps, ((max_new - min_new) / (max_old - min_old)) * (value - min_old) + min_new)

        car_gen_steps = np.rint(car_gen_steps)  # round every value to int -> effective steps when a car will be generated

        # produce the file for cars generation, one car per line
        with open("intersection/STLs.rou.xml", "w") as routes:
            print("""<routes>
            <vType accel="1.0" decel="4.5" id="standard_car" length="5.0" minGap="2.5" maxSpeed="25" sigma="0.5" />

            


            <route id="r_0" edges="E4"/>

            <route id="r_1" edges="E4 -E3"/>

            <route id="r_2" edges="E4 -E5"/>

            <route id="r_3" edges="E3 -E5"/>

            <route id="r_4" edges="E3 -E4"/>

            <route id="r_5"  edges="E5 -E4"/>

            <route id="r_6" edges="E5 -E3"/>""", file=routes)

            for car_counter, step in enumerate(car_gen_steps):
                straight_or_turn = np.random.uniform()
                if straight_or_turn < 0.75:  # choose direction: straight or turn - 75% of times the car goes straight
                    route_straight = np.random.randint(1, 5)  # choose a random source & destination
                    if route_straight == 1:
                        print('    <vehicle id="r_0_%i" type="standard_car" route="r_0" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_straight == 2:
                        print('    <vehicle id="r_1_%i" type="standard_car" route="r_1" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_straight == 3:
                        print('    <vehicle id="r_3_%i" type="standard_car" route="r_3" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    else:
                        print('    <vehicle id="r_4_%i" type="standard_car" route="r_4" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                else:  # car that turn -25% of the time the car turns
                    route_turn = np.random.randint(1, 9)  # choose random source source & destination
                    if route_turn == 1:
                        print('    <vehicle id="r_5_%i" type="standard_car" route="r_5" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_turn == 2:
                        print('    <vehicle id="r_6_%i" type="standard_car" route="r_6" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_turn == 3:
                        print('    <vehicle id="r_1_%i" type="standard_car" route="r_1" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_turn == 4:
                        print('    <vehicle id="r_2_%i" type="standard_car" route="r_2" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_turn == 5:
                        print('    <vehicle id="r_3_%i" type="standard_car" route="r_3" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_turn == 6:
                        print('    <vehicle id="r_4_%i" type="standard_car" route="r_4" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_turn == 7:
                        print('    <vehicle id="r_5_%i" type="standard_car" route="r_5" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)
                    elif route_turn == 8:
                        print('    <vehicle id="r_6_%i" type="standard_car" route="r_6" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, step), file=routes)

            print("</routes>", file=routes)
