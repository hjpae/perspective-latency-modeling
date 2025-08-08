"""
Test run v1.0 - for rendering behaviors 
: behav rendering run for khepera-based agents (forager and hunter).

This code evaluates a single individual's function by loading stored NN weights
and logs input, hidden, and output neuron activations.

- The testing environment is the same as the training environment (object placement logic unchanged).

Jan 2025, written by kadi
"""

#%% preambles
from controller import Supervisor
import numpy as np
import random
import math
import csv
import os

# =========================
# Hyperparameters
GENERATION = 300  # Set your target generation here
RANK = 0         # Set your desired rank index here
# =========================

# NN hyperparameters
NUM_INPUT = 4  # distance sensor vectorized input(2) + camera vectorized input(2)
NUM_HIDDEN = 8
NUM_OUTPUT = 2  # left motor speed, right motor speed

# Agent task parameters
SCORE_GREEN = 300  # score for encountering a green object
PENALTY_RED = 400  # penalty for encountering a red object
DISTANCE_THRESHOLD = 0.1  # threshold for proximity detection

REVISIT_PENALTY = 1  # penalty for revisiting the same location
REVISIT_THRESHOLD = 0.02  # threshold for revisited location detection

MAX_STEPS = 1000  # extended steps for better rendering

# Paths
GENOME_CSV = "/home/hjpae/hoss-iit/animats/log/genome_forager_v2.1_set1.csv"
ACTIVATION_LOG_DIR = "/home/hjpae/hoss-iit/animats/log/behavior_logs_hunter/"

# Create activation log directory if it doesn't exist
if not os.path.exists(ACTIVATION_LOG_DIR):
    os.makedirs(ACTIVATION_LOG_DIR)

#%% NeuralNetwork Class
class NeuralNetwork:
    """
    RNN with:
      - Input dim = 4
      - Hidden dim = 8
      - Output dim = 2

    Weights:
      - W_xh: (4,8)
      - W_hh: (8,8)
      - W_hy: (8,2)

    Activation:
      - tanh on hidden and output
    """
    def __init__(self, weights):
        # Slicing points / total 112 nodes
        idx_w_xh_end = NUM_INPUT * NUM_HIDDEN  # 4*8 = 32
        idx_w_hh_end = idx_w_xh_end + (NUM_HIDDEN * NUM_HIDDEN)  # +64=96
        idx_w_hy_end = idx_w_hh_end + (NUM_HIDDEN * NUM_OUTPUT)  # +16=112

        # input->hidden
        self.W_xh = weights[:idx_w_xh_end].reshape(NUM_INPUT, NUM_HIDDEN)
        # hidden->hidden
        self.W_hh = weights[idx_w_xh_end:idx_w_hh_end].reshape(NUM_HIDDEN, NUM_HIDDEN)
        # hidden->output
        self.W_hy = weights[idx_w_hh_end:idx_w_hy_end].reshape(NUM_HIDDEN, NUM_OUTPUT)

        self.hidden_state = np.zeros(NUM_HIDDEN, dtype=float)

    def reset_state(self):
        self.hidden_state[:] = 0.0

    def forward(self, x_input):
        """
        RNN step:
         h(t+1) = tanh(x(t)*W_xh + h(t)*W_hh)
         y(t+1) = tanh(h(t+1)*W_hy)
        """
        h_in = np.dot(x_input, self.W_xh) + np.dot(self.hidden_state, self.W_hh)
        self.hidden_state = np.tanh(h_in)
        out = np.dot(self.hidden_state, self.W_hy)
        return np.tanh(out)

#%% ObjectRandomizer Class
class ObjectRandomizer:
    """
    Object randomizing logic is unchanged from the training environment.
    """
    def __init__(self, supervisor, min_distance=0.1, robot_distance=0.1, arena_size=0.4):
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

        self.min_distance = min_distance
        self.robot_distance = robot_distance
        self.arena_size = arena_size
        self.positions = []

        self.robot_node = self.supervisor.getFromDef("ROBOT")
        self.robot_position_field = self.robot_node.getField("translation")
        self.robot_rotation_field = self.robot_node.getField("rotation")

    def is_too_close(self, x, y):
        for pos in self.positions:
            dist = math.sqrt((x - pos[0])**2 + (y - pos[1])**2)
            if dist < self.min_distance:
                return True
        robot_position = self.robot_position_field.getSFVec3f()
        dist_r = math.sqrt((x-robot_position[0])**2 + (y-robot_position[1])**2)
        if dist_r < self.robot_distance:
            return True
        return False

    def randomize_positions(self):
        self.positions=[]
        for obj in self.green_objects + self.red_objects:
            while True:
                x = random.uniform(-self.arena_size,self.arena_size)
                y = random.uniform(-self.arena_size,self.arena_size)
                z = 0.05
                if not self.is_too_close(x,y):
                    self.positions.append((x,y))
                    obj.getField("translation").setSFVec3f([x,y,z])
                    break

    def check_proximity_and_relocate(self):
        robot_position = self.robot_position_field.getSFVec3f()
        new_positions=[]
        for obj in self.green_objects + self.red_objects:
            obj_position=obj.getField("translation").getSFVec3f()
            dist=math.sqrt((robot_position[0]-obj_position[0])**2 + (robot_position[1]-obj_position[1])**2)
            if dist<self.robot_distance:
                print(f"Relocating {obj.getDef()} due to proximity ({dist:.3f} m).")
                while True:
                    x = random.uniform(-self.arena_size,self.arena_size)
                    y = random.uniform(-self.arena_size,self.arena_size)
                    z = 0.05
                    if not self.is_too_close(x,y):
                        new_positions.append((x,y))
                        obj.getField("translation").setSFVec3f([x,y,z])
                        break
            else:
                new_positions.append((obj_position[0],obj_position[1]))
        self.positions=new_positions

    def run(self):
        self.randomize_positions()
        while self.supervisor.step(self.time_step)!=-1:
            self.check_proximity_and_relocate()

#%% Agent Class
class Agent:
    """
    The agent logic is also identical to the training environment, so that behavior is consistent.
    """
    def __init__(self):
        self.supervisor = Supervisor()
        self.time_step = int(self.supervisor.getBasicTimeStep())

        # motors
        self.left_motor = self.supervisor.getDeviceByIndex(1)
        self.right_motor= self.supervisor.getDeviceByIndex(3)
        self.left_motor.setPosition(float('inf'))
        self.right_motor.setPosition(float('inf'))
        self.left_motor.setAvailableTorque(0.5)
        self.right_motor.setAvailableTorque(0.5)

        # sensors
        self.left_distance_sensors=[self.supervisor.getDevice(f"ds{i}") for i in range(3)]
        self.right_distance_sensors=[self.supervisor.getDevice(f"ds{i}") for i in range(3,6)]
        for s in (self.left_distance_sensors + self.right_distance_sensors):
            s.enable(self.time_step)

        self.camera=self.supervisor.getDevice("camera")
        self.camera.enable(self.time_step)

        self.robot_node=self.supervisor.getFromDef("ROBOT")
        self.robot_translation_field=self.robot_node.getField("translation")
        self.robot_rotation_field=self.robot_node.getField("rotation")
        self.initial_position=[0,0,0.01]
        self.initial_rotation=[0,0,1,1.5708]

        self.DISTANCE_THRESHOLD=DISTANCE_THRESHOLD
        self.PENALTY_RED=PENALTY_RED
        self.SCORE_GREEN=SCORE_GREEN

        self.score=0
        self.penalty=0
        self.revisit_penalty=0
        self.proximity_green=0

        self.object_randomizer=ObjectRandomizer(self.supervisor)

        self.REVISIT_THRESHOLD=REVISIT_THRESHOLD
        self.visited_positions={}
        self.idle_count=0

    def reset_metrics(self):
        self.score=0
        self.penalty=0
        self.proximity_green=0
        self.revisit_penalty=0
        self.visited_positions={}
        self.idle_count=0

    def reset_robot(self):
        self.robot_translation_field.setSFVec3f(self.initial_position)
        self.robot_rotation_field.setSFRotation(self.initial_rotation)
        self.supervisor.simulationResetPhysics()
        self.left_motor.setVelocity(0)
        self.right_motor.setVelocity(0)
        self.reset_metrics()
        self.object_randomizer.randomize_positions()

    def get_grid_index(self,pos):
        gx=int(math.floor(pos[0]/self.REVISIT_THRESHOLD))
        gy=int(math.floor(pos[1]/self.REVISIT_THRESHOLD))
        return (gx,gy)

    def process_distance_sensors(self):
        max_sensor_value=1023.0
        min_sensor_value=0.0
        left_vals=[s.getValue() for s in self.left_distance_sensors]
        right_vals=[s.getValue() for s in self.right_distance_sensors]
        left_norm=[(v-min_sensor_value)/(max_sensor_value-min_sensor_value) for v in left_vals]
        right_norm=[(v-min_sensor_value)/(max_sensor_value-min_sensor_value) for v in right_vals]
        return np.mean(left_norm), np.mean(right_norm)

    def vectorize_camera(self):
        image=self.camera.getImage()
        w=self.camera.getWidth()
        h=self.camera.getHeight()
        left_green=0; right_green=0
        left_red=0; right_red=0

        for x in range(w):
            for y in range(h):
                r=self.camera.imageGetRed(image,w,x,y)
                g=self.camera.imageGetGreen(image,w,x,y)
                b=self.camera.imageGetBlue(image,w,x,y)
                if x<(w//2):
                    if g>1.1*r and g>1.1*b:
                        left_green+=1
                    elif r>1.1*g and r>1.1*b:
                        left_red+=1
                else:
                    if g>1.1*r and g>1.1*b:
                        right_green+=1
                    elif r>1.1*g and r>1.1*b:
                        right_red+=1
        total_pix=(w//2)*h
        left_diff=(left_green-left_red)/total_pix
        right_diff=(right_green-right_red)/total_pix
        scale=5.0
        l_int=max(-1,min(1,left_diff*scale))
        r_int=max(-1,min(1,right_diff*scale))
        return l_int,r_int

    def vectorize_sensors(self):
        ld,rd=self.process_distance_sensors()
        lc,rc=self.vectorize_camera()
        return np.array([ld, rd, lc, rc])

    def check_proximity(self,ld,rd,lc,rc):
        greens=["GREEN_OBJECT_1","GREEN_OBJECT_2","GREEN_OBJECT_3"]
        reds=["RED_OBJECT_1","RED_OBJECT_2","RED_OBJECT_3"]
        rob_pos=self.robot_translation_field.getSFVec3f()
        for obj_name in (greens+reds):
            node=self.supervisor.getFromDef(obj_name)
            if not node: continue
            obj_pos=node.getField("translation").getSFVec3f()
            dist=math.sqrt((rob_pos[0]-obj_pos[0])**2+(rob_pos[1]-obj_pos[1])**2)
            if dist<self.DISTANCE_THRESHOLD:
                if obj_name in greens:
                    print(f"Green {obj_name} +{SCORE_GREEN}")
                    self.score+=SCORE_GREEN
                else:
                    print(f"Red {obj_name} +{PENALTY_RED}")
                    self.penalty+=PENALTY_RED
                return True
        return False

    def step_reward(self):
        greens=["GREEN_OBJECT_1","GREEN_OBJECT_2","GREEN_OBJECT_3"]
        rob_pos=self.robot_translation_field.getSFVec3f()
        min_d=float('inf')
        for g in greens:
            nd=self.supervisor.getFromDef(g)
            if not nd: continue
            gpos=nd.getField("translation").getSFVec3f()
            d=math.sqrt((rob_pos[0]-gpos[0])**2+(rob_pos[1]-gpos[1])**2)
            if d<min_d:
                min_d=d
        threshold=0.2
        if min_d<threshold:
            return (threshold-min_d)/threshold*5
        return 0.0

    def run_simulation(self, nn, generation=0, MAX_STEPS=MAX_STEPS, individual_idx=None, log_activations=True):
        self.reset_metrics()
        nn.reset_state()
        steps=0
        idle_steps=0
        epsilon=1e-4

        # randomize objects as in training
        self.object_randomizer.randomize_positions()

        activation_log=[] if log_activations else None

        while self.supervisor.step(self.time_step)!=-1:
            orientation=self.robot_node.getOrientation()
            up_z=orientation[8]
            if up_z<0.707:
                self.robot_translation_field.setSFVec3f(self.initial_position)
                self.robot_rotation_field.setSFRotation(self.initial_rotation)
                self.supervisor.simulationResetPhysics()

            sensor_in=self.vectorize_sensors()
            ld,rd,lc,rc=sensor_in
            out=nn.forward(sensor_in)

            ls=out[0]*10.0
            rs=out[1]*10.0
            if abs(ls)<2.0 and abs(rs)<2.0:
                ls=random.uniform(3.0,6.0)*random.choice([-1,1])
                rs=random.uniform(3.0,6.0)*random.choice([-1,1])

            if max(abs(lc),abs(rc))<epsilon and min(ld,rd)>epsilon:
                idle_steps+=1
                self.idle_count+=1
                if idle_steps>10:
                    ls=random.uniform(4.0,8.0)*random.choice([-1,1])
                    rs=random.uniform(4.0,8.0)*random.choice([-1,1])
            else:
                idle_steps=0

            self.left_motor.setVelocity(ls)
            self.right_motor.setVelocity(rs)

            rpos=self.robot_translation_field.getSFVec3f()
            gidx=(int(math.floor(rpos[0]/REVISIT_THRESHOLD)),
                  int(math.floor(rpos[1]/REVISIT_THRESHOLD)))
            self.visited_positions[gidx]=self.visited_positions.get(gidx,0)+1
            self.revisit_penalty+=(REVISIT_PENALTY+self.visited_positions[gidx]*0.7)

            if self.check_proximity(ld,rd,lc,rc):
                self.object_randomizer.check_proximity_and_relocate()

            self.proximity_green+=self.step_reward()

            if log_activations:
                record={
                    'step':steps,
                    'input':sensor_in.copy(),
                    'hidden_state':nn.hidden_state.copy(),
                    'output':out.copy()
                }
                activation_log.append(record)

            if steps%200==0:
                print(f"[Gen {generation}|Ind {individual_idx} Step:{steps}]"
                      f" Score={self.score}, PScore={self.proximity_green}, Pen={self.penalty}, "
                      f"Rpen={self.revisit_penalty}, Idle={self.idle_count}")

            steps+=1
            if steps>=MAX_STEPS:
                break

        print("Simulation ended (max steps).")
        print(f"[Gen {generation}|Ind {individual_idx}] Score={self.score}, ProxScore={self.proximity_green}, "
              f"Penalty={self.penalty}, Rpen={self.revisit_penalty}, Idle={self.idle_count}")

        if activation_log is not None:
            return steps, self.score, self.penalty, self.revisit_penalty, self.proximity_green, self.idle_count, activation_log
        else:
            return steps, self.score, self.penalty, self.revisit_penalty, self.proximity_green, self.idle_count


#%% Helper to load a specific generation & rank from the CSV
def load_genome(csv_path, generation, rank):
    """
    Loads a single genome from csv.
    """
    with open(csv_path,'r',newline='') as f:
        reader=csv.DictReader(f)
        for row in reader:
            try:
                row_gen=int(row["Generation"])
                row_rank=int(row["Rank"])
                if row_gen==generation and row_rank==rank:
                    wvals=[float(row[k]) for k in row if k.startswith("W_")]
                    wvals=np.array(wvals,dtype=float)
                    print(f"Loaded Genome for Gen={generation}, Rank={rank}, count={len(wvals)}")
                    return wvals
            except (KeyError, ValueError) as e:
                print(e)
                continue
    print(f"No genome found for Gen={generation} Rank={rank}")
    return None

#%% main
def main():
    # Hard-coded generation and rank
    generation = GENERATION
    rank = RANK

    # 1) load genome
    w=load_genome(GENOME_CSV, generation, rank)
    if w is None:
        print("Genome not found. Exiting.")
        return

    # 2) create NN
    nn=NeuralNetwork(w)

    # 3) create agent
    agent=Agent()

    # 4) run sim with activation logging
    print(f"=== Running single simulation for Gen={generation}, Rank={rank} ===")
    sim_result=agent.run_simulation(nn, generation=generation, individual_idx=rank, log_activations=True)

    if len(sim_result)==7:
        steps, score, penalty, rpen, pg, idle, activation_log = sim_result
    else:
        print("Unexpected return from run_simulation.")
        return

    # 5) Save activation log
    activation_csv=os.path.join(ACTIVATION_LOG_DIR, f"gen_{generation}_rank_{rank}_behavior.csv")
    with open(activation_csv,'w',newline='') as cf:
        writer=csv.writer(cf)
        head=["Step","Input1","Input2","Input3","Input4"]+[f"H{i}" for i in range(NUM_HIDDEN)]+["Output1","Output2"]
        writer.writerow(head)
        for rec in activation_log:
            st=rec['step']
            ins=rec['input']
            hid=rec['hidden_state']
            outs=rec['output']
            row=[st]+list(ins)+list(hid)+list(outs)
            writer.writerow(row)

    print(f"Wrote activation data to {activation_csv}")
    print("Behavior rendering run completed.")

if __name__=="__main__":
    main()
