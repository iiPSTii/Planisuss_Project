# Planisuss Ecosystem Simulation

## Overview
Planisuss is a Python-based simulation of a simplified ecosystem designed to study
population dynamics and ecological interactions over time.  
The ecosystem is modeled as a **2D grid world** populated by three entities:

- **Vegetob** – vegetation that grows daily  
- **Erbast** – herbivores that consume vegetation  
- **Carviz** – carnivores that hunt herbivores  

The simulation models growth, consumption, hunting, movement, aging, and reproduction,
and provides real-time graphical visualization of spatial and temporal dynamics.

---

## Project Objectives
- Simulate the dynamic evolution of a simplified ecosystem  
- Study interactions and balance between vegetation, herbivores, and carnivores  
- Analyze population trends over time  
- Visualize ecological dynamics through maps and plots  

---

## Simulation Setup and Constraints
- Grid size: **100 × 100**
- Simulation duration: **500 days**
- Water cells (inhabitable): **15%**
- Vegetation growth: **+1 per day per cell (capped)**
- Maximum energy per individual: **100**
- Maximum age: **100 days**
- Population caps per cell:
  - Erbast: **1000**
  - Carviz: **100**
- Probabilistic spawning mechanism, independent of direct parent presence  

---

## Model Assumptions
- Cells do not interact directly, but individuals can move to adjacent cells  
- Each individual has independent attributes (energy, age, lifespan, social behavior)  
- The environment contains static components (water, grid structure) and dynamic ones
  (vegetation and animals)  
- Random initialization and stochastic behavior simulate natural variability  

---

## Daily Simulation Cycle
Each simulated day is divided into five phases:

1. Vegetation growth and overwhelm control  
2. Movement driven by resources and social behavior  
3. Grazing (Erbast consume vegetation)  
4. Fight and hunt (Carviz dominance and predation)  
5. Aging, death, and reproduction  

---

## Implementation and Design Choices
The project follows an **object-oriented programming (OOP)** approach to ensure modularity
and clarity.

### Main Components
- **Erbast class**: herbivores characterized by energy, age, lifespan, and social behavior  
- **Carviz class**: carnivores with hunting and dominance interactions  
- **Cell class**: represents a grid cell containing vegetation and populations  
- **World class**: manages the grid and executes all ecological phases  

Helper functions are used to cap values and compute valid neighboring cells efficiently.

---

## Visualization
The simulation produces real-time visual output using `matplotlib`, including:

- Composite RGB map of the ecosystem  
- Vegetob heatmap  
- Erbast heatmap  
- Carviz heatmap  
- Logarithmic population trend plot  


---

## Results
- Initial population fluctuations stabilize after approximately **60–70 days**  
- Erbast and Carviz populations grow and reach equilibrium  
- Vegetob initially decreases, then stabilizes relative to consumer populations  
- Spatial maps highlight migration patterns and density variations  

These results indicate that the system reaches a stable ecological balance.

---

## Tools and Libraries
- **Python 3.11**
- NumPy
- Matplotlib
- Random
- Sys
