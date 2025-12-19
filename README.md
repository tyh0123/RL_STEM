# Interpretable Digital Twins for Autonomous STEM Aberration Correction

This repository contains a proof-of-concept framework for **machine-learning-assisted aberration correction in scanning transmission electron microscopy (STEM)**. The project develops a corrector digital twin that integrates **LLM-based log parsing, symbolic regression, and reinforcement learning** to enable faster, more stable, and reproducible aberration correction with reduced reliance on expert operators.

## Overview

Achieving sub-ångström resolution in STEM requires precise tuning of lens aberrations. In practice, aberration correction is a nonlinear, strongly coupled, and operator-dependent process: adjusting one aberration coefficient often perturbs multiple others, leading to unstable convergence and repeated trial-and-error tuning.

This project introduces an **interpretable digital twin framework** that learns aberration–response relationships directly from real experimental logs and uses them to autonomously optimize correction strategies.

<p align="center">
  <img src="figures/AI_TEM.png" alt="Figure 1: AI-based digital twin framework for STEM aberration correction" width="90%">
</p>

*Figure 1: AI-based digital twin framework integrating LLM-based experimental log parsing, symbolic regression-derived coupling models, a corrector simulator, and reinforcement learning-based optimization in a closed loop with the physical microscope.*

## Framework Components

The framework consists of four tightly coupled modules:

1. **LLM-Based Log Parsing**  
   Human-generated STEM aberration correction logs are automatically parsed and standardized using a large language model. The parser extracts time-ordered aberration coefficients, correction actions, and convergence behavior into structured trajectories suitable for downstream learning.

2. **Symbolic Regression for Aberration Coupling**  
   A sparsity-promoting symbolic regression method (SINDy-style) is applied to the parsed trajectories to identify **nonlinear and cross-coupled response relationships** between corrector settings and aberration evolution. The resulting analytical expressions provide interpretable models of corrector behavior.

3. **Corrector Digital Twin Simulator**  
   A lightweight aberration corrector simulator embeds the learned symbolic response functions. Given an initial aberration state and a correction action, the simulator predicts the updated aberration coefficients, optionally incorporating noise and actuation constraints to mimic experimental conditions.

4. **Reinforcement Learning–Based Optimization**  
   Aberration correction is formulated as a sequential decision-making problem and solved using a Proximal Policy Optimization (PPO) agent. The agent outputs multi-parameter correction actions simultaneously, learning a robust closed-loop policy under cross-coupled dynamics and stochastic disturbances.

## Repository Structure

## Repository Structure and Code Description

This repository implements an interpretable digital twin framework for autonomous STEM aberration correction, integrating symbolic regression, a corrector simulator, and reinforcement learning. The main components and scripts are described below.

### Data

- **A1_actions.csv**  
- **C1_actions.csv**  

Parsed aberration correction trajectories extracted from real STEM alignment logs.  
Each file contains time-ordered aberration coefficients and corresponding correction actions for a specific aberration mode, serving as training data for symbolic regression.

### End-to-End Workflow

- **corrector_play.ipynb**  

Jupyter notebook demonstrating the complete end-to-end pipeline.  
Walks through symbolic regression training, construction of the corrector digital twin, reinforcement learning training, and evaluation, serving as a reproducible reference workflow.


- **results/**  

Directory containing completed training runs and evaluation results from symbolic regression and reinforcement learning experiments.
### Symbolic Regression (Aberration Coupling Models)

- **sr_sindy_v2_A1_complete.py**  
- **sr_sindy_v2_A1_complete_C1.py**  
- **sr_sindy_v2_C1_complete.py**  

Symbolic regression training scripts based on sparsity-promoting SINDy-style methods.  
These scripts learn nonlinear and cross-coupled response relationships between corrector actions and aberration evolution from experimental correction trajectories. The learned analytical models provide interpretable approximations of corrector behavior and are embedded into the digital twin.

### Digital Twin Simulator

- **CorrectorPlayEnv_v4.py**  

Core digital twin environment for the aberration corrector.  
Implements the learned symbolic response models, correction action constraints, stochastic disturbances, and reward formulation. This environment serves as the simulation backend for reinforcement learning training and evaluation.

### Reinforcement Learning

- **train_ppo_v4.py**  

Reinforcement learning training script using Proximal Policy Optimization (PPO).  
Trains a policy that outputs multi-parameter aberration correction actions simultaneously, accounting for nonlinear and cross-coupled correction dynamics in the digital twin environment.

- **evaluate_v4.py**  

Evaluation and testing script for trained reinforcement learning policies.  
Assesses convergence speed, stability, and correction performance under different initial aberration conditions in the digital twin.

### Graphical User Interface

- **Play_UI_v4.py**  

Graphical user interface for the corrector digital twin.  
Provides an interactive control panel that mimics experimental corrector software, enabling manual testing, visualization of aberration evolution, and closed-loop interaction with the simulator.

### Utilities

- **requirements.txt**  

List of Python dependencies required to run the symbolic regression, digital twin simulator, reinforcement learning training, and GUI.

