# Simulating Subjective Inference by Perspective Conditioning 

<p align="center">
  <img src="assets/steppe_hunter_test_g250_pes.gif" width="300"/>
  <img src="assets/steppe_hunter_test_gen250_opt.gif" width="300"/>
</p>

<p align="center">
  <b>Agent A (Cautious)</b> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp <b>Agent B (Bold)</b>
</p>



## Overview
This project explores how artificial agents can modulate their behavior not solely based on objective environmental rewards, but through *subjective interpretive stance*, namely **perspective**.  
I model these perspectives as latent variables that bias the agent's evaluation of reward structures. Importantly, perspectives are fluid, and can change or drift in response to environmental circumstances. This idea provides a minimal computational foundation for simulating lifelong developmental change and world model adjustment through adaptive inference.

## Core Idea
Most AI systems optimize reward directly and objectively. In contrast, this system shows:
- Agents **do not perceive rewards directly**.
- Instead, each agent holds a **latent structure *(perspective)*** that **biases** its internal evaluation of risk and reward.
- This bias (`risk_bias`) **adapts over time** based on agentic interaction with the environment.

## Simulation Idea
### General environmental setup
The simulation takes place in a 50cm x 50cm 2D arena. While Webots simulates 3D physics, the z-axis is unused.  
The agent spawns at the center, and two types of objects are randomly spawned throughout the space: 
- **Red objects** offer high rewards but are risky (20% chance of penalty).
- **Green objects** are safe (no penalty) but yield lower rewards.  

Objects are re-spawned at new random positions upon collision with the agent. 

### Phase 1: Learning the Behavior under Fixed perspective 
Two agents are initialized with different fixed perspectives:
- **Agent A ("Cautious")**: avoids reward-high but risky red objects. `risk_bias = -0.9`
- **Agent B ("Bold")**: more actively seeks out the same red objects, discounting its risk. `risk_bias = 0.9`

Each agent evolves its own policy/behavior pattern using the Genetic Evolutionary Algorithm (GEA) applied to an 8-node Recurrent Neural Network (RNN). The fitness function of GEA includes task scores, spatial efficiency, and motion efficiency.  
After convergence, the evolved weight vectors (genomes) are collected for use in Phase 2.  

### Phase 2: Observe the Perspective Drift in Changing Environments  
Using the evolved policies from Phase 1, agents are now placed in altered environments:
- **Scenario "Drought"**: Green objects are rare.  
- **Scenario "Hard Hunt"**: Red penalty probability increases; green becomes more rewarding.  
- **Scenario "Easy Prey"**: Red becomes safer, but also less rewarding.  

As the environment shifts, the agents' perspectives and behaviors are supposed to shift as follows:  
- Agent A may become more risk-seeking in "Drought"
- Agent B may become more cautious in "Hard Hunt"
- Both agents may converge in "Easy Prey"

This demonstrates **perspectival adaptation**, where agents revise their interpretation of reward structure based on surprise.


## Simulation Setup 
- **Engine**: Python + [Webots](https://cyberbotics.com/)
- **Dependencies**: Webots R2023b, Python 3.8.10

### Architecture and Learning mechanism
- **Agent controller**: 8-node Recurrent Neural Network (RNN)
  - **Input**: 4D, distance sensors (left, right) + color vision (left, right)
  - **Output**: 2D, motor velocity (left, right)
  
- **Learning method**: Genetic Evolutionary Algorithm (GEA)
  - Fitness is based on scoring success & penalty, proximity to the objects, and movement efficiency (penalizes revisiting and being idle): 
    ```python
    fitness = total_score + proximity_bonus - collision_penalty - revisit_penalty - idle_penalty
    ```
  - Risk evaluation is modulated by the agent's `risk_bias`.

- **Evolution constraints**:
  - 60 individual episodes per 1 generation
  - 15 elite parents -> 50 children / 45 remaining parents -> 10 remaining children
  - Mutation: multi-point crossover with 0.05 mutation rate 
  - 800 time steps per episode
  - Typically takes 250-300 generation runs for the fitness function to converge

### Perspective Drift
Agents begin with fixed `risk_bias` values. After Phase 1, agents retain their previously evolved RNN weights as initial policies. In Phase 2, the agent's internal perspective (`risk_bias`) becomes plastic, adapting over time through experience.  
This `risk_bias` directly modifies the **fitness evaluation** via a biased interpretation of reward outcomes. As perspective shifts, the fitness landscape changes, which in turn steers further evolution of behavior (i.e. the genome continues evolving under new interpretive criteria).

- Drift is triggered by **mismatch between expected and observed reward**, quantified through surprisal:

    ```python
    error = observed_reward - expected_reward
    surprisal = -log2(P(observed_reward))
    adjustment = learning_rate * sign(error) * tanh(surprisal) # lr * vector * magnitude
    risk_bias += adjustment
    ```

Through this structure, the agents adapt their internal interpretation of risk by revising their perspective. This perspective then reshapes how behavior is evaluated.

### Environment (Phase 1)
| Object     | Reward     | Penalty     |
|------------|------------|-------------|
| Green (3)  | +200       | None        |
| Red (3)    | +800       | -400 (20%)  |

- 3 red and 3 green objects are randomly placed in the arena at all times.  

### Environment (Phase 2, "Drought" Scenario) 
| Object     | Reward     | Penalty     |
|------------|------------|-------------|
| Green (1)  | +200       | None        |
| Red (5)    | +800       | -400 (20%)  |

- Only 1 green object is spawned, while the remaining 5 objects (out of 6 total) are all red.
- Reward and penalty values remain unchanged.


### Environment (Phase 2, "Hard Hunt" Scenario) 
| Object     | Reward     | Penalty     |
|------------|------------|-------------|
| Green (3)  | +400       | None        |
| Red (3)    | +800       | -500 (50%)  |
- Red objects are now riskier, with both a higher penalty and a higher penalty probability.
- The reward for green objects is increased.
- The object counts remain the same. 

### Environment (Phase 2, "Easy Prey" Scenario) 
| Object     | Reward     | Penalty     |
|------------|------------|-------------|
| Green (3)  | +200       | None        |
| Red (3)    | +300       | -200 (10%)  |
- Red objects now yield lower rewards and also carry a smaller penalty with lower risk.
- Green object settings and the object counts remain the same. 

## Perspective Architecture

The `Perspective` class encodes how an agent subjectively interprets reward structure. It defines a biased belief distribution over possible red outcomes, and updates the `risk_bias` based on surprisal.


```python
class Perspective:
    def __init__(self, risk_bias, learning_rate=0.1):
        self.risk_bias = np.clip(risk_bias, -1.0, 1.0)
        self.learning_rate = learning_rate

    def red_outcome_distribution(self, score_red, penalty_red, base_penalty_chance):
        shift = 0.2 * self.risk_bias
        p_penalty = np.clip(base_penalty_chance - shift, 0.01, 0.99)
        p_reward = 1 - p_penalty
        return {score_red: p_reward, -penalty_red: p_penalty}

    def expected_red_reward(self, score_red, penalty_red, base_penalty_chance):
        dist = self.red_outcome_distribution(score_red, penalty_red, base_penalty_chance)
        return sum([val * prob for val, prob in dist.items()])

    def update(self, observed_reward, score_red, penalty_red, base_penalty_chance):
        dist = self.red_outcome_distribution(score_red, penalty_red, base_penalty_chance)
        p_obs = dist.get(observed_reward, 0.01)
        surprisal = -np.log2(p_obs)

        expected = self.expected_red_reward(score_red, penalty_red, base_penalty_chance)
        error = observed_reward - expected
        direction = np.sign(error)

        adjustment = self.learning_rate * direction * np.tanh(surprisal)
        self.risk_bias += adjustment
        self.risk_bias = np.clip(self.risk_bias, -1.0, 1.0)
```

## Interim Results
### Behavior comparison of Phase 1
Watch how agents with different perspectives behave differently in the same environment:
- **Agent A (Cautious)** avoids red, favors green. 
- **Agent B (Bold)** aggressively targets red despite penalties.  

The demo GIF can be found at the top of this page. For additional demo videos, see the /assets/ folder.

