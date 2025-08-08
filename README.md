# Simulating Subjective Inference by Perspective Conditioning 

<p align="center">
  <img src="assets/steppe_hunter_test_g250_pes.gif" width="300"/>
  <img src="assets/steppe_hunter_test_gen250_opt.gif" width="300"/>
</p>

<p align="center">
  <b>Agent A (risk-averse): "forages" the green </b> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp <b>Agent B (risk-taking): "hunts" the red </b>
</p>

## Table of Contents
- [Overview](#overview)
- [Core Idea](#core-idea)
- [Simulation Idea](#simulation-idea)
- [Simulation Setup](#simulation-setup)
- [Perspective Architecture](#perspective-architecture)
- [Interim Results](#interim-results)
- [Development Status](#development-status)
  - [Research Notes](#research-notes)

## Overview
This project explores how artificial agents can modulate their behavior not solely based on objective environmental rewards, but through *subjective interpretive stance*, namely **perspective**. I model these perspectives as **latent variables** that **bias the agent's evaluation of reward structures**. Importantly, **perspectives are fluid**, and can change or drift in response to environmental circumstances. This idea provides a minimal computational foundation for simulating lifelong developmental change and world model adjustment through adaptive inference.

## Core Idea
Most AI systems optimize reward directly and objectively. In contrast, this system shows:
- Agents **do not perceive rewards directly**.
- Instead, each agent holds a **latent structure *(perspective)*** that **biases** its internal evaluation of risk and reward.
- This bias (`risk_bias`) **adapts over time** based on agentic interaction with the environment.

### Why This Matters 
Much of AI behavior optimization operates under the assumption of an objective, externally defined reward structure. Yet real-world agents (especially humans) interpret and evaluate value through inherently **subjective interpretation**. Phenomenology—from the transcendental tradition to the embodiment paradigm—has long emphasized that all conscious mental processes, or cognitive acts, are already imbued with subjective interpretation of the world [(Pae 2025 preprint, under revision)](https://osf.io/preprints/psyarxiv/pzrx8_v1). This means we never encounter the world "as-is" in a purely objective sense; rather, we experience it through the interpretive lens of a situated subject. Subjectivity is inevitable for every conscious being.  

Such intuition is formalized in this project as a dynamic latent structure, namely, the **agent's "perspective"** - which modulates how observed outcomes are interpreted as success (under a bold/optimistic point of view) or failure (under a cautious/pessimistic point of view). This can be seen as the simplest and most primitive form of a **world model**, or even an **instantiation of a phenomenological artifact** *in silico*, in the sense that the agent is engaging in its own subjective interpretation of the world. 

Importantly, this interpretive structure is not fixed. It evolves, drifts, and reorganizes in light of the agent's perspective-changing process, allowing the agent to refine not only its policies but the evaluative lens through which it interprets experience. In this sense, perspective drift offers a minimal scaffold for simulating **lifelong development**, **perspective-taking**, and **social alignment**.  

By embedding subjective interpretation into the policy/behavior optimization process, this approach opens a new way to study:
- How **qualitative stance** (e.g. boldness, cautiousness) emerges and changes, 
- How agents might **misalign or realign** themselves - not due to objective shifts in reward but due to **subjective adaptation**,
- and How we might simulate aspects of **first-person cognition** beyond standard policy learning.  

Ultimately, this work builds a conceptual bridge between **developmental neuroscience** and **computational neurophenomenology**. It may also inform new paradigms in AI alignment research, particularly in those that seek agent alignment through world-interpreting process - i.e. not merely *what the agent does*, but *how it comes to see and feel* what it does. 


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
    adjustment = sign(error) * tanh(surprisal) * learning_rate  # vector * magnitude * lr
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

---

## Development Status 
This project is currently under active development. All core mechanisms are implemented and functional, and further experiments and visualizations (especially from Phase 2 and beyond) are ongoing. Stay tuned for the updates. 

### Research Notes 
- How appropriate is it to model perspective drift/update using Bayesian prediction error minimization as suggested by the active inference framework? When a human perceives a concept or entity, can we really say that their "shift in perspective" emerges solely from active inference? After all, perspective change resembles a developmental process, often involving qualitative shifts in self-interpretation. I suspect there's a more nuanced way to capture this.  
  - Increasing the modality of the environment might be crucial, as richer sensorimotor dynamics allow for deeper embodiment and experiential grounding.
  - One possible extension is to implement meta-level latent-latent processes that influence the evolution of perspective. Since metacognition is suggested to play a central role in [adult development](http://onesystemonevoice.com/resources/Cook-Greuter+9+levels+paper+new+1.1$2714+97p$5B1$5D.pdf), [learning](https://www.nature.com/articles/s41539-021-00089-5), and [world model building](https://arxiv.org/abs/2411.13537), refactoring the perspective model hierarchically may help simulate meta-level paradigm shifts in the world model.  
- Improving the agent's internal architecture is also a key direction. While RNN/LSTM architectures have their own virtues due to their simplicity, they may be too limited to capture consciousness. Exploring architectures that could implement [SOHMs](https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2020.00030/full) (Self-Organizing Harmonic Modes) may offer a more appropriate framework.  
- If one truly aims to model conscious-like phenomena, then pre-reflective self-acquaintance (the felt presence of the body prior to cognitive reflection) becomes essential. This inevitably ties to the simulation environment, which must be rich and sophisticated enough to support agents not as disembodied brains, but as embodied entities who can recognize their own givenness through embodied interaction.  
- The idea of learning a world model in latent space bears structural resemblance to [Dreamer](https://github.com/google-research/dreamer). A comparative analysis between my model and Dreamer might yield valuable insights into both their similarities and foundational differences.

### Hence, Future Directions:  
1. Understanding Perspective Drift 
   - Analyze how perspective drift unfolds over time by tracking its trajectory alongside fitness convergence in evolving agents.
   - Move beyond simple prediction-error-based updates toward richer models of interpretive change.
 
2. Meta-Structure & Hierarchical Perspective Modeling
   - Introduce a meta-level layer that governs how perspectives themselves are interpreted and selected.
   - Explore "latent-latent" architectures where perspective drift is modulated by higher-order inference.
   - Consider modeling qualitative shifts (something like developmental stage transitions) as structural reconfigurations rather than mere parameter updates.

3. Agent Architecture Redesign
   - Investigate more expressive agent architectures beyond standard RNNs: consider biologically inspired models like [SOHMs](https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2020.00030/full) to support richer internal dynamics, synchrony, and emergent resonance.

4. Environment & Embodiment
   - Replace Webots with more modality-rich environments like [Common Grounds](https://commongrounds.ai/), [Metta-grid](https://github.com/Metta-AI/metta), or custom-built PettingZoo-based environments.
   - Design agents with genuine bodily presence to enable pre-reflective self-acquaintance, which is a foundation for embodied consciousness.

5. Comparative Analysis with Dreamer
   - Compare and contrast this system with [Dreamer](https://github.com/google-research/dreamer), particularly:
     - Both rely on latent dynamics ... while Dreamer models objective environmental dynamics vs. this project focuses on subjective evaluative structures. 
   - Investigate how each framework deals with planning, representation, and self-modification.
