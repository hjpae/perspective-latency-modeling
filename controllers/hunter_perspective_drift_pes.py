"""
hunter - "aggressive carnivore" with perspective settings. 
# "drifting" perspective aka developmental perspective modeling added. 

Aim of the agent is to: 
1. collide (<10cm) to as many red objects 
2. under given time limit (800 steps), 
3. have preference to red over green, but sometimes red gives penalty 

Aug 2025 written by kadi 
"""

#%% preambles
from controller import Robot, Supervisor, Motor, DistanceSensor, Camera
import numpy as np
import random
import math

import csv
import os

# NN hyperparameters
NUM_INPUT = 4  # distance sensor vectorized input(2) + camera vectorized input(2)
NUM_OUTPUT = 2  # left motor speed, right motor speed
NUM_HIDDEN = 8

# EA parameters
POPULATION_SIZE = 60
MUTATION_RATE = 0.05
MAX_GENERATIONS = 601 # 300-601
MAX_STEPS = 800 

# Agent task parameters
RISK_BIAS = 0.9 # [-1, 1] ... [risk-averse---neutral---risk-taking]
SCORE_GREEN = 200 # score for encountering a green object
SCORE_RED = 800 # score for encountering a red object
PENALTY_RED = 400 # penalty for encountering a red object

PENALTY_RED_CHANCE = 0.2 # 20% chance
DISTANCE_THRESHOLD = 0.1  # threshold for proximity detection

REVISIT_PENALTY = 1  # penalty for revisiting the same location 
REVISIT_THRESHOLD = 0.02  # threshold for revisited location detection

# Logging and saving the data 
DATA_DIR = "/home/hjpae/hoss-iit/animats/log/genome_hunter_v3.1_pes.csv"

def log_generation_to_csv(population, fitness_scores, generation, filename = DATA_DIR):
    """
    population: shape = (POPULATION_SIZE, total_weights_dim)
    fitness_scores: shape = (POPULATION_SIZE,)
    generation: int (current generation)
    DATA_DIR: str (.csv file path)
    """
    # descending order fitness
    sorted_indices = np.argsort(fitness_scores)[::-1]

    file_exists = os.path.exists(filename)
    with open(filename, mode='a', newline='') as csvfile:
        writer = csv.writer(csvfile)

        if not file_exists:
            header = ["Generation", "Rank", "Fitness"]
            header += [f"W_{i}" for i in range(population.shape[1])]
            writer.writerow(header)

        for rank, idx in enumerate(sorted_indices):
            fit_val = fitness_scores[idx]
            genome = population[idx]
            row = [generation, rank, f"{fit_val:.3f}"] + genome.tolist()
            writer.writerow(row)

def load_generation_from_csv(filename, target_generation, 
                             param_count=112,
                             pop_size=60):
    """
    load population(gene) of certain generation, from logged .csv
    output shape: (pop_size, param_count)

    log.csv header should look like:
    [Generation, Rank, Fitness, W_0, W_1, ..., W_{param_count-1}]
    """

    loaded = []
    with open(filename, 'r', newline='') as f:
        reader = csv.reader(f)
        header = next(reader)  # the first line: [Generation, Rank, Fitness, W_0, W_1, ..., W_{param_count-1}]

        for row in reader:
            gen = int(row[0])  # generation
            if gen == target_generation:
                # weights = row[3:] ~ num of param 
                w_strs = row[3 : 3 + param_count]
                w_vals = list(map(float, w_strs))
                loaded.append(w_vals)

    arr = np.array(loaded, dtype=float)
    if arr.shape[0] != pop_size:
        print(f"[WARNING] Found {arr.shape[0]} individuals for generation={target_generation}, "
              f"expected {pop_size}. Check .csv or pop_size mismatch.")
    return arr

#%% NN Class
class NeuralNetwork:
    """
    RNN with:
      - Input dim = 4
      - Hidden dim = 8
      - Output dim = 2
    Weights:
      - W_xh: (4, 8)   (input to hidden)
      - W_hh: (8, 8)   (hidden to hidden)
      - W_hy: (8, 2)   (hidden to output)

    Activation:
      - tanh on hidden and output
    """
    def __init__(self, weights):
        # slicing points / total 112 nodes 
        idx_w_xh_end = NUM_INPUT * NUM_HIDDEN # 4*8
        idx_w_hh_end = idx_w_xh_end + (NUM_HIDDEN * NUM_HIDDEN) # 8*8
        idx_w_hy_end = idx_w_hh_end + (NUM_HIDDEN * NUM_OUTPUT) # 8*2

        # input to hidden weight, W_xh
        self.W_xh = weights[:idx_w_xh_end].reshape(NUM_INPUT, NUM_HIDDEN)

        # hidden to hidden weight, W_hh
        self.W_hh = weights[idx_w_xh_end:idx_w_hh_end].reshape(NUM_HIDDEN, NUM_HIDDEN)

        # hidden to output weight, W_hy
        self.W_hy = weights[idx_w_hh_end:idx_w_hy_end].reshape(NUM_HIDDEN, NUM_OUTPUT)

        # hidden layer state vector 
        self.hidden_state = np.zeros(NUM_HIDDEN, dtype=float)

    def reset_state(self):
        """
        Resets the hidden state before every simulation run(episode) begins. 
        """
        self.hidden_state = np.zeros(NUM_HIDDEN, dtype=float)

    def forward(self, x_input):
        """
        one step of RNN consists of:
         h(t+1) = tanh( x(t)*W_xh + h(t)*W_hh )
         y(t+1) = tanh( h(t+1)*W_hy )

        x_input: shape=(4,) (sensor input)
        return: shape=(2,) (motor output - tanh recommended)
        """

        # update the hidden state
        h_in = np.dot(x_input, self.W_xh) + np.dot(self.hidden_state, self.W_hh)
        self.hidden_state = np.tanh(h_in) # [-1 1] ...relu() makes Phi calculation tricky

        # calculate the output
        out = np.dot(self.hidden_state, self.W_hy)
        out = np.tanh(out) # [-1 1]

        return out, self.hidden_state.copy()

#%% Object randomizer Class (originally from a separate controller)
class ObjectRandomizer:
    def __init__(self, supervisor, min_distance=0.1, robot_distance=0.1, arena_size=0.4):
        # try out different variables
        self.supervisor = supervisor
        self.time_step = int(supervisor.getBasicTimeStep())

        self.green_objects = [
            self.supervisor.getFromDef("GREEN_OBJECT_1"),
            self.supervisor.getFromDef("GREEN_OBJECT_2"),
            self.supervisor.getFromDef("GREEN_OBJECT_3")
        ]

        self.red_objects = [
            self.supervisor.getFromDef("RED_OBJECT_1"),
            self.supervisor.getFromDef("RED_OBJECT_2"),
            self.supervisor.getFromDef("RED_OBJECT_3")
        ]

        # parameters
        self.min_distance = min_distance  # minimum distance between objects
        self.robot_distance = robot_distance  # threshold distance for detecting proximity
        self.arena_size = arena_size  # half the arena size (x, y range will be -arena_size to arena_size)
        self.positions = []  # keep track of object positions

        # robot position
        self.robot_node = self.supervisor.getFromDef("ROBOT")
        self.robot_position_field = self.robot_node.getField("translation")
        self.robot_rotation_field = self.robot_node.getField("rotation")

    def is_too_close(self, x, y):
        """Check if the position is too close to existing objects or the robot."""
        for pos in self.positions:
            distance = math.sqrt((x - pos[0]) ** 2 + (y - pos[1]) ** 2)
            if distance < self.min_distance:
                return True

        robot_position = self.robot_position_field.getSFVec3f()
        distance_to_robot = math.sqrt((x - robot_position[0]) ** 2 + (y - robot_position[1]) ** 2)
        if distance_to_robot < self.robot_distance:
            return True

        return False

    def randomize_positions(self):
        """Randomly place objects in the arena while avoiding overlaps."""
        self.positions = []
        for obj in self.green_objects + self.red_objects:
            while True:
                x = random.uniform(-self.arena_size, self.arena_size)
                y = random.uniform(-self.arena_size, self.arena_size)
                z = 0.05
                if not self.is_too_close(x, y):
                    self.positions.append((x, y))
                    obj.getField("translation").setSFVec3f([x, y, z])
                    break

    def check_proximity_and_relocate(self):
        """
        Check if objects are within proximity of the robot and relocate them if necessary.
        """
        robot_position = self.robot_position_field.getSFVec3f()
        new_positions = []

        for obj in self.green_objects + self.red_objects:
            obj_position = obj.getField("translation").getSFVec3f()
            distance = math.sqrt(
                (robot_position[0] - obj_position[0]) ** 2 +
                (robot_position[1] - obj_position[1]) ** 2
            )

            if distance < self.robot_distance:
                print(f"Relocating {obj.getDef()} due to proximity ({distance:.3f} m).")
                while True:
                    x = random.uniform(-self.arena_size, self.arena_size)
                    y = random.uniform(-self.arena_size, self.arena_size)
                    z = 0.05
                    if not self.is_too_close(x, y):
                        new_positions.append((x, y))
                        obj.getField("translation").setSFVec3f([x, y, z])
                        break
            else:
                new_positions.append((obj_position[0], obj_position[1]))

        self.positions = new_positions  # update position list

    def run(self):
        """
        Main loop to randomize positions and continuously check proximity.
        """
        self.randomize_positions()

        while self.supervisor.step(self.time_step) != -1:
            self.check_proximity_and_relocate()

#%% perspective class (new from v3.0!!)
class Perspective:
    """
    A hyperprior-like modulator of the fitness function.
    Influences agent's preference toward risky red reward.

    Perspective is a subjective interpretation of risk/reward tradeoff.
    risk_bias is dynamically updated based on experience.
    """
    def __init__(self, risk_bias, learning_rate=0.1):
        """
        risk_bias: float in [-1.0, 1.0]
        -1.0 = highly risk-averse (penalty-sensitive)
        0.0 = neutral
        +1.0 = highly risk-seeking (penalty-tolerant)
        """
        self.risk_bias = np.clip(risk_bias, -1.0, 1.0)  # how agent perceives red risk
        self.learning_rate = learning_rate  # developmental plasticity 
        self.history = []

    def red_outcome_distribution(self, score_red, penalty_red, base_penalty_chance):
        """
        Returns agent's *biased belief* about red reward outcome distribution.
        Output: dict of {reward_val: probability}
        """
        # base probs
        p_penalty = base_penalty_chance
        p_reward = 1 - base_penalty_chance

        # bias shift (risk_bias in [-1, 1])
        shift = 0.2 * self.risk_bias  # tunable hyperparam
        p_penalty_biased = np.clip(p_penalty - shift, 0.01, 0.99)
        p_reward_biased = 1 - p_penalty_biased

        return {
            score_red: p_reward_biased,
            -penalty_red: p_penalty_biased
        }

    def expected_red_reward(self, score_red=SCORE_RED, penalty_red=PENALTY_RED, penalty_chance=PENALTY_RED_CHANCE):
        """
        Calculates biased expected value of red reward.
        This modifies the effective value used in fitness.
        """
        dist = self.red_outcome_distribution(score_red, penalty_red, penalty_chance)
        return sum([val * prob for val, prob in dist.items()])

    def adjust_fitness(self, red_score, green_score, penalty_chance, penalty_red, score_red, reward_green):
        ev_red = self.expected_red_reward(score_red, penalty_red, penalty_chance)
        adjusted_red = red_score * ev_red / score_red

        return adjusted_red + green_score  # green is always added unmodified

    def update(self, observed_reward, expected_reward):
        """
        Update the risk_bias based on prediction error.
        If observed < expected, agent becomes more cautious.
        If observed > expected, agent becomes more risk-tolerant.
        
        observed_reward: the scalar outcome received (e.g., +800 or -400)
        """
        dist = self.red_outcome_distribution(score_red, penalty_red, penalty_chance)
        p_obs = dist.get(observed_reward, 0.01) # avoid log(0)
        surprisal = -np.log2(p_obs) # the magnitude of the surprise (how drastic the drift happens)

        expected = self.expected_red_reward(score_red, penalty_red, penalty_chance)
        error = observed_reward - expected
        direction = np.sign(error) # the vector of the surprise (cautious vs. risk-tolerant)

        adjustment = self.learning_rate * direction * np.tanh(surprisal)
        self.risk_bias += adjustment
        self.risk_bias = np.clip(self.risk_bias, -1.0, 1.0)
        self.history.append(self.risk_bias)


#%% GeneticAlgorithm Class
class GeneticAlgorithm:
    def __init__(self):
        # random weight to create initial population
        self.population = np.random.uniform(
            -0.3, 0.3, # be mindful when setting the initial value!!! 
            (POPULATION_SIZE, 
                (NUM_INPUT * NUM_HIDDEN) 
                + (NUM_HIDDEN * NUM_HIDDEN) 
                + (NUM_HIDDEN * NUM_OUTPUT))
        )
        self.perspective = Perspective(RISK_BIAS)

    def evaluate_fitness(self, agent, generation):
        fitness_scores = []
        for i, weights in enumerate(self.population):
            nn = NeuralNetwork(weights)
            # reset the environment and weights for each individual
            nn.reset_state()
            agent.reset_robot()

            # run the simulation and collect fitness data
            steps, red_score, green_score, red_collision_penalty, revisit_penalty, proximity_red, idle_count = agent.run_simulation(nn, generation=generation, individual_idx=i)
            fitness = self.calculate_fitness(steps, red_score, green_score, revisit_penalty, proximity_red, idle_count, red_collision_penalty)
            fitness_scores.append(fitness)
        return np.array(fitness_scores)

    def calculate_fitness(self, steps, red_score, green_score, revisit_penalty, proximity_red, idle_count, red_collision_penalty):
        
        # perspective-conditioned reward
        total_score = self.perspective.adjust_fitness(
            red_score=red_score,
            green_score=green_score,
            penalty_chance=PENALTY_RED_CHANCE,
            penalty_red=PENALTY_RED,
            score_red=SCORE_RED,
            reward_green=SCORE_GREEN
        )

        proximity_bonus = proximity_red * 0.3
        collision_penalty = red_collision_penalty * 10
        revisit_penalty = revisit_penalty * 0.8
        idle_penalty = idle_count * 12 if red_score > 3 else idle_count * 8

        fitness = total_score + proximity_bonus - collision_penalty - revisit_penalty - idle_penalty
        return fitness

    def select_and_reproduce(self, fitness_scores):
        top_parents_count = 15
        top_children_count = 50
        other_children_count = 10

        # select top 15 parents
        top_indices = np.argsort(fitness_scores)[::-1][:top_parents_count] # descending order (larger fitness the better)
        top_parents = [self.population[i] for i in top_indices]

        # select remaining parents
        remaining_indices = np.argsort(fitness_scores)[::-1][top_parents_count:] # descending order!!
        remaining_parents = [self.population[i] for i in remaining_indices]

        # generate 50 children from top 15 parents
        new_population = []
        for _ in range(top_children_count):
            parent1, parent2 = random.sample(top_parents, 2)
            crossover_mask = np.random.randint(2, size=parent1.shape)
            child = parent1 * crossover_mask + parent2 * (1 - crossover_mask)
            child += np.random.uniform(-MUTATION_RATE, MUTATION_RATE, len(child))
            new_population.append(child)

        # generate 10 children from the remaining parents
        for _ in range(other_children_count):
            parent1, parent2 = random.sample(remaining_parents, 2)
            crossover_mask = np.random.randint(2, size=parent1.shape)
            child = parent1 * crossover_mask + parent2 * (1 - crossover_mask)
            child += np.random.uniform(-MUTATION_RATE, MUTATION_RATE, len(child))
            new_population.append(child)

        self.population = np.array(new_population[:POPULATION_SIZE])


#%% Agent Class
class Agent:
    def __init__(self):
        #self.robot = Robot()
        self.supervisor = Supervisor()
        self.time_step = int(self.supervisor.getBasicTimeStep())

        # motors
        self.left_motor = self.supervisor.getDeviceByIndex(1)  # left motor
        self.right_motor = self.supervisor.getDeviceByIndex(3)  # right motor
        self.left_motor.setPosition(float('inf'))
        self.left_motor.setAvailableTorque(0.5)
        self.right_motor.setPosition(float('inf'))
        self.right_motor.setAvailableTorque(0.5)

        # distance sensors (split into left and right groups)
        self.left_distance_sensors = [self.supervisor.getDevice(f"ds{i}") for i in range(3)] # ds0, ds1, ds2
        self.right_distance_sensors = [self.supervisor.getDevice(f"ds{i}") for i in range(3, 6)] # ds3, ds4, ds5
        for sensor in self.left_distance_sensors + self.right_distance_sensors:
            sensor.enable(self.time_step)

        # camera for color detection
        self.camera = self.supervisor.getDevice("camera")
        self.camera.enable(self.time_step)

        # position field for reset
        self.robot_node = self.supervisor.getFromDef("ROBOT")
        self.robot_translation_field = self.robot_node.getField("translation")
        self.robot_rotation_field = self.robot_node.getField("rotation")
        self.initial_position = [0, 0, 0.01]  # default: center of the arena
        self.initial_rotation = [0, 0, 1, 1.5708] # default angle (90degree = 1.5708rad)

        # task-specific parameters
        self.DISTANCE_THRESHOLD = DISTANCE_THRESHOLD
        self.PENALTY_RED = PENALTY_RED 
        self.SCORE_RED = SCORE_RED
        self.SCORE_GREEN = SCORE_GREEN

        # score and penalty metrics, NOT params! (**connected with GeneticAlgorithm class**)
        self.red_score = 0
        self.green_score = 0
        self.red_collision_penalty = 0
        self.revisit_penalty = 0
        self.proximity_red = 0

        # object location reset (**connected with ObjectRandomizer class**)
        self.object_randomizer = ObjectRandomizer(self.supervisor)

        # check visited positions 
        self.REVISIT_THRESHOLD = REVISIT_THRESHOLD
        self.visited_positions = {} # {grid_index: visited_pos}
        self.idle_count = 0 

    def reset_metrics(self):
        # resets score and penalty for each simulation run
        self.red_score = 0
        self.green_score = 0
        self.red_collision_penalty = 0
        self.proximity_red = 0
        self.revisit_penalty = 0
        self.visited_positions = {}
        self.idle_count = 0 

    def reset_robot(self):
        # resets robot and object location for each simulation run
        self.robot_translation_field.setSFVec3f(self.initial_position)
        self.robot_rotation_field.setSFRotation(self.initial_rotation)
        self.supervisor.simulationResetPhysics()
        self.left_motor.setVelocity(0)
        self.right_motor.setVelocity(0)
        self.reset_metrics()
        self.object_randomizer.randomize_positions()

    def get_grid_index(self, position):
        """
        Get the robot's position into grid index. (for location revisit check) 
        position: [x, y, z]
        return: tuple (grid_x, grid_y)
        """
        grid_x = int(math.floor(position[0] / self.REVISIT_THRESHOLD))
        grid_y = int(math.floor(position[1] / self.REVISIT_THRESHOLD))
        return (grid_x, grid_y)

    def process_distance_sensors(self):
        """
        Separately processes left and right distance sensors.
        Then, normalize the value between 0~10, 
        Then returns the average for left and right groups.
        """
        max_sensor_value = 1023.0  # from lookupTable
        min_sensor_value = 0.0     # from lookupTable
        scale_factor = 1.0        # normalize to range [0, 1]

        left_values = [sensor.getValue() for sensor in self.left_distance_sensors]
        right_values = [sensor.getValue() for sensor in self.right_distance_sensors]

        left_values_normalized = [
            ((value - min_sensor_value) / (max_sensor_value - min_sensor_value)) * scale_factor
            for value in left_values
        ]
        right_values_normalized = [
            ((value - min_sensor_value) / (max_sensor_value - min_sensor_value)) * scale_factor
            for value in right_values
        ]

        # take the average of normalized values
        left_avg = np.mean(left_values_normalized)
        right_avg = np.mean(right_values_normalized)

        #print(f"Distances - Left: {left_avg:.2f}, Right: {right_avg:.2f}")

        return left_avg, right_avg

    def vectorize_camera(self):
        """
        Processes camera image and calculates normalized color intensity for left and right fields of view.
        """
        image = self.camera.getImage()
        width = self.camera.getWidth()
        height = self.camera.getHeight()

        left_green, right_green, left_red, right_red = 0, 0, 0, 0

        # process each pixel
        for x in range(width):
            for y in range(height):
                r = self.camera.imageGetRed(image, width, x, y)
                g = self.camera.imageGetGreen(image, width, x, y)
                b = self.camera.imageGetBlue(image, width, x, y)

                if x < width // 2: # left half of the camera view
                    # if g > r and g > b:
                    #     left_green += 1
                    # elif r > g and r > b:
                    #     left_red += 1
                    # when green intensity is sufficiently higher than red and blue
                    if g > (1.1 * r) and g > (1.1 * b): # threshold = 1.1
                        left_green += 1
                    # when red intensity is sufficiently higher than green and blue
                    elif r > (1.1 * g) and r > (1.1 * b):
                        left_red += 1

                else: # right half of the camera view
                    # if g > r and g > b:
                    #     right_green += 1
                    # elif r > g and r > b:
                    #     right_red += 1
                    # when green intensity is sufficiently higher than red and blue
                    if g > (1.1 * r) and g > (1.1 * b):
                        right_green += 1
                    # when red intensity is sufficiently higher than green and blue
                    elif r > (1.1 * g) and r > (1.1 * b):
                        right_red += 1

        # normalize color intensities (green - red) / total pixels per side
        total_pixels = (width // 2) * height

        left_diff = (left_green - left_red) / total_pixels
        right_diff = (right_green - right_red) / total_pixels

        # scaling factor to improve sensitivity
        scaling_factor = 5.0  
        left_intensity = max(-1, min(1, left_diff * scaling_factor))
        right_intensity = max(-1, min(1, right_diff * scaling_factor))

        #print(f"Camera Intensities - Left: {left_intensity:.3f}, Right: {right_intensity:.3f}")
        return left_intensity, right_intensity

    def vectorize_sensors(self):
        """
        combine left/right distance sensors and camera inputs into a single input vector for the NN.
        Here, NN input consists of 4 nodes. 
        """
        left_distance, right_distance = self.process_distance_sensors()
        left_camera, right_camera = self.vectorize_camera()

        #print(f"Inputs - Left Distance: {left_distance:.3f}, Right Distance: {right_distance:.3f}, "
        #     f"Left Camera: {left_camera:.3f}, Right Camera: {right_camera:.3f}")

        return np.array([left_distance, right_distance, left_camera, right_camera])

    def distance_to_nearest_red(self):
        """
        returns the minimum distance from the robot to any red object.
        (for hunter v2.4)
        """
        red_names = ["RED_OBJECT_1", "RED_OBJECT_2", "RED_OBJECT_3"]
        robot_pos = self.robot_translation_field.getSFVec3f()

        min_dist_r = float('inf')
        for rname in red_names:
            rnode = self.supervisor.getFromDef(rname)
            if rnode is None:
                continue
            rpos = rnode.getField("translation").getSFVec3f()
            dist = math.sqrt( (robot_pos[0]-rpos[0])**2 + (robot_pos[1]-rpos[1])**2 )
            if dist < min_dist_r:
                min_dist_r = dist

        return min_dist_r

    def distance_to_nearest_green(self):
        """
        returns the minimum distance from the robot to any green object.
        (for hunter v2.4)
        """
        green_names = ["GREEN_OBJECT_1", "GREEN_OBJECT_2", "GREEN_OBJECT_3"]
        robot_pos = self.robot_translation_field.getSFVec3f()

        min_dist_g = float('inf')
        for gname in green_names:
            gnode = self.supervisor.getFromDef(gname)
            if gnode is None:
                continue
            gpos = gnode.getField("translation").getSFVec3f()
            dist = math.sqrt( (robot_pos[0]-gpos[0])**2 + (robot_pos[1]-gpos[1])**2 )
            if dist < min_dist_g:
                min_dist_g = dist

        return min_dist_g

    def check_proximity(self, left_distance, right_distance, left_camera, right_camera):
        """
        Detect proximity to green or red objects based on distance and camera inputs.
        """
        robot_position = self.robot_translation_field.getSFVec3f()
        green_objects = ["GREEN_OBJECT_1", "GREEN_OBJECT_2", "GREEN_OBJECT_3"]
        red_objects = ["RED_OBJECT_1", "RED_OBJECT_2", "RED_OBJECT_3"]

        for obj_name in green_objects + red_objects:
            obj_node = self.supervisor.getFromDef(obj_name)
            obj_position = obj_node.getField("translation").getSFVec3f()

            distance = np.sqrt((robot_position[0] - obj_position[0])**2 +
                               (robot_position[1] - obj_position[1])**2)

            if distance < self.DISTANCE_THRESHOLD:
                if obj_name in red_objects:
                    self.red_score += self.SCORE_RED
                    print(f"Collided RED {obj_name}, score+{self.SCORE_RED}.")

                    if random.random() < PENALTY_RED_CHANCE:
                        self.red_collision_penalty += 1
                        print(f"Penalty incurred! {PENALTY_RED} applied.")

                elif obj_name in green_objects:
                    self.green_score += self.SCORE_GREEN
                    print(f"Collided GREEN {obj_name}, +{self.SCORE_GREEN} points.")

                return True

        return False

    def step_reward(self):
        """
        Reward (+) for getting closer to the red obj with each step.
        Example: distance d <= 0.25, (25cm ... arena size: 100cm) 
            Reward = (0.25 - d) / 0.25 * 5 (-> up to 5 points per step)
            if distance d >= 0.25, reward = 0 
        """
        red_names = ["RED_OBJECT_1", "RED_OBJECT_2", "RED_OBJECT_3"]
        robot_pos = self.robot_translation_field.getSFVec3f()

        # search min distance among all green objs
        min_dist = float('inf')
        for rname in red_names:
            rnode = self.supervisor.getFromDef(rname)
            if rnode is None: 
                continue
            rpos = rnode.getField("translation").getSFVec3f()
            dist = math.sqrt((robot_pos[0]-rpos[0])**2 + (robot_pos[1]-rpos[1])**2)
            if dist < min_dist:
                min_dist = dist

        threshold = 0.2 
        if min_dist < threshold:
            # more closer, the greater reward
            reward = (threshold - min_dist) / threshold * 5 # up to 5 points per step
            return reward
        else:
            return 0.0

    def run_simulation(self, nn, generation=0, MAX_STEPS=MAX_STEPS, individual_idx=None):
        # Step 0: initialize metrics and counters
        idle_steps = 0
        self.reset_metrics()
        nn.reset_state()
        steps = 0
        epsilon = 1e-4  # idle threshold for camera inputs
        hidden_state_log = []

        # make sure the object randomizer starts correctly
        self.object_randomizer.randomize_positions()

        while self.supervisor.step(self.time_step) != -1:
            # check if robot is filpped
            orientation = self.robot_node.getOrientation() # 3×3 rotation matrix (row-major)
            up_z = orientation[8]  # x-axis filp, 0.707rad = ~45degree
            if up_z < 0.707:
                self.robot_translation_field.setSFVec3f(self.initial_position)
                self.robot_rotation_field.setSFRotation(self.initial_rotation)
                self.supervisor.simulationResetPhysics()

            # Step 1: Read sensor inputs
            sensor_inputs = self.vectorize_sensors()
            left_distance, right_distance, left_camera, right_camera = sensor_inputs

            # Step 2: NN computes motor outputs
            motor_output, hidden_state = nn.forward(sensor_inputs)
            hidden_state_log.append(hidden_state.copy())

            scaled_left_speed = motor_output[0] * 10.0  # scale output to [-5, 5]
            scaled_right_speed = motor_output[1] * 10.0

            # if output is close to 0, assign small default values to prevent stalling
            if abs(scaled_left_speed) < 2.0 and abs(scaled_right_speed) < 2.0:
                #print("Low motor output detected. Applying aggressive scaling.")
                scaled_left_speed = random.uniform(3.0, 6.0) * random.choice([-1, 1])
                scaled_right_speed = random.uniform(3.0, 6.0) * random.choice([-1, 1])

            #print(f"NN Inputs: {sensor_inputs}")

            # Step 3: Idle detection
            # idle is triggered when camera input is inactive and the robot is not near any object
            if max(abs(left_camera), abs(right_camera)) < epsilon and min(left_distance, right_distance) > epsilon:
                idle_steps += 1
                self.idle_count += 1
            else:
                idle_steps = 0  # reset idle counter if inputs are active

            # Step 4: Set motor speeds
            self.left_motor.setVelocity(scaled_left_speed)
            self.right_motor.setVelocity(scaled_right_speed)
            #print(f"NN Motor Output: {motor_output}")
            # print the inputs and outputs, for debugging purpose 
            if steps % 200 == 0:
                print(f"Inputs: {sensor_inputs}, Outputs: {motor_output}")
                print(f"Individual {individual_idx}, Gen {generation}, Step: {steps}, IdleSteps: {self.idle_count}, RScore: {self.red_score}, GScore: {self.green_score}, PScore: {self.proximity_red}, Penalty: {self.red_collision_penalty}, RPenalty: {self.revisit_penalty}")
            steps += 1

            # Step 5: check overlapping visits
            current_pos = self.robot_translation_field.getSFVec3f()  # [x, y, z]
            grid_index = self.get_grid_index(current_pos)

            if grid_index in self.visited_positions:
                self.visited_positions[grid_index] += 1
            else:
                self.visited_positions[grid_index] = 1 # 1p per each time

            revisit_penalty_increment = REVISIT_PENALTY + (self.visited_positions[grid_index] * 0.7)
            self.revisit_penalty += revisit_penalty_increment

            # Step 6: check proximity and relocate objects if necessary 
            proximity_triggered = self.check_proximity(left_distance, right_distance, left_camera, right_camera)
            if proximity_triggered:
                self.object_randomizer.check_proximity_and_relocate()

            # Step 7: Give step reward
            step_prox_reward = self.step_reward()
            self.proximity_red += step_prox_reward
            #print(f"approaching red, proximity score: {self.proximity_red}")

            # check if max steps reached
            if steps >= MAX_STEPS:
                #print(f"Max steps reached: {MAX_STEPS}")
                break

        # Step 8: End individual run if max steps reached
        print("Simulation ended: Maximum steps reached.")
        print(f"Individual {individual_idx}, Gen {generation}, Final IdleSteps: {self.idle_count}, Final RScore: {self.red_score}, Final GScore: {self.green_score}, Final PScore: {self.proximity_red}, Final Penalty: {self.red_collision_penalty}, Final RPenalty: {self.revisit_penalty}")
        hidden_state_log = np.array(hidden_state_log)
        return steps, self.red_score, self.green_score, self.red_collision_penalty, self.revisit_penalty, self.proximity_red, self.idle_count


#%% Simulation
def main():
    agent = Agent()
    ga = GeneticAlgorithm()

    for generation in range(MAX_GENERATIONS):
        print(f"\n=== Generation {generation} ===")
        fitness_scores = ga.evaluate_fitness(agent, generation)
        ga.select_and_reproduce(fitness_scores)

        best_fitness = np.max(fitness_scores) # larger score the better!!!!! 
        print(f"Generation {generation}: Best Fitness = {best_fitness}")

        if generation % 10 == 0:  # log the weights per every 10th gen, including 0th gen 
            log_generation_to_csv(
                population=ga.population,
                fitness_scores=fitness_scores,
                generation=generation,
                filename=DATA_DIR 
            )
            print(f"Logged generation {generation} to CSV.")

        agent.reset_robot()

def main_load():
    agent = Agent()
    ga = GeneticAlgorithm()

    # Step 1. load generation from log.csv
    csv_file = DATA_DIR
    generation_to_load = 300
    param_count = 112  
    pop_size    = 60   
    loaded_pop = load_generation_from_csv(csv_file, generation_to_load, 
                                          param_count=param_count,
                                          pop_size=pop_size)
    if loaded_pop.shape[0] == pop_size:
        ga.population = loaded_pop
        print(f"Successfully loaded generation={generation_to_load} population from CSV.")
    else:
        print("Error: mismatch or not found generation in CSV. Check logs.")
        return

    # Step 2. run the simulation again starting from loaded generation 
    start_gen = generation_to_load + 1  # =491
    for generation in range(start_gen, MAX_GENERATIONS):
        print(f"\n=== Generation {generation} ===")
        fitness_scores = ga.evaluate_fitness(agent, generation)
        ga.select_and_reproduce(fitness_scores)

        best_fitness = np.max(fitness_scores)
        print(f"Generation {generation}: Best Fitness = {best_fitness}")

        # log the weights per every 10th gen
        if generation % 10 == 0:
            log_generation_to_csv(ga.population, fitness_scores, generation, filename=DATA_DIR)
            print(f"Logged generation {generation} to CSV.")

        agent.reset_robot()

# if __name__ == "__main__":
#     main()

if __name__ == "__main__":
    main_load()
