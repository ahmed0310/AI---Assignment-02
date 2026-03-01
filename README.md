# Dynamic Pathfinding Agent — AI2002 Assignment 02 (Q6)

## 📌 Overview

This project implements a **Dynamic Pathfinding Agent** that navigates a grid-based environment using **Informed Search Algorithms**.  
The agent is capable of **real-time re-planning** when obstacles appear dynamically while it is moving.

The application is built using **Python and Pygame** and visually demonstrates the behavior and performance of:
- **A\* Search**
- **Greedy Best-First Search (GBFS)**

This project fulfills **Question 06** of **AI-2002 (Artificial Intelligence) – Assignment 02**.

---

## 🎯 Features

### ✅ Environment
- Dynamic grid size (rows × columns)
- Fixed **Start Node** (top-left)
- Fixed **Goal Node** (bottom-right)
- Random maze generation with configurable obstacle density
- Interactive wall placement & removal using mouse

### ✅ Algorithms Implemented
- **A\* Search**  
  Uses `f(n) = g(n) + h(n)` (optimal with admissible heuristics)
- **Greedy Best-First Search (GBFS)**  
  Uses `f(n) = h(n)` (fast but not optimal)

### ✅ Heuristics
- Manhattan Distance
- Euclidean Distance  
(Selectable from GUI before running the search)

### ✅ Dynamic Mode
- Obstacles spawn randomly while the agent is moving
- Agent detects blocked paths
- Immediate **re-planning from current position**
- Optimized to avoid unnecessary full resets

### ✅ Visualization
- Frontier (Open List)
- Explored (Closed List)
- Final Path
- Agent movement animation

### ✅ Real-Time Metrics
- Nodes Visited
- Path Cost
- Execution Time (ms)

---

## 🖥️ GUI Controls

| Control | Description |
|------|------------|
| Algorithm Dropdown | Select A\* or Greedy BFS |
| Heuristic Dropdown | Select Manhattan or Euclidean |
| Run Search | Starts the selected algorithm |
| Reset Grid | Clears walls and paths |
| Random Map | Generates random obstacles |
| Dynamic Mode | Enables obstacle spawning |

---

## 🧠 Project Structure

.
├── main.py # Complete application (grid, algorithms, GUI, agent)
├── README.md # Project documentation




> **Note:**  
All components are merged into a single file (`main.py`) as required.

---

## 🛠️ Installation & Setup

### 1️⃣ Requirements
- Python **3.9+**
- Pygame

### 2️⃣ Install Dependencies
```bash
pip install pygame
```

### 3️⃣ Run the Application
```bash
python main.py
```


