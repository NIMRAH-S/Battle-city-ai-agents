# Battle City AI 🎮

A Python and Pygame-based recreation of the classic arcade game *Battle City*, featuring advanced Artificial Intelligence. 

Unlike the original game which relied on simple randomized logic, this project serves as an AI sandbox where different tank classes are driven by distinct, academic AI architectures ranging from Simple Reflex Agents to Adversarial Minimax algorithms.

## Features

### 🧠 Advanced Enemy AI
The game features 5 distinct enemy classes, each implementing a unique AI architecture:
* **Basic Tank (Simple Reflex Agent):** Uses simple IF-THEN rules and **BFS (Breadth-First Search)** pathfinding. It recalculates paths dynamically and will aggressively clear destructible walls in its path.
* **Fast Tank (Goal-Based Agent):** Uses **Greedy Best-First Search**. It strictly focuses on destroying the player's Eagle but is prone to getting stuck in local minima (designed to showcase the weaknesses of Greedy search vs A*).
* **Armor Tank (Model-Based Reflex Agent):** Maintains internal state (hit points). It uses **A* Pathfinding** to attack, but upon taking critical damage, transitions into a retreating state—running a BFS search to find the nearest indestructible Steel wall for cover.
* **Power Tank (Utility-Based Agent):** Calculates dynamic utility scores on the fly to decide whether it is more advantageous to attack the player or rush the Eagle, navigating via **A***.
* **Boss Tank (Adversarial Agent):** A multi-phase boss utilizing the **Minimax Algorithm with Alpha-Beta Pruning**. It evaluates game states up to 4 turns into the future to predict player movement, prioritize cover, and calculate lethal attack angles.

### 🗺️ Dynamic Environment
* **Destructible Terrain:** Brick walls can be destroyed by bullets, dynamically altering the AI's pathfinding mesh in real-time.
* **Forests:** Tanks hiding in forest tiles become invisible to the player.
* **Pixel-Perfect Physics:** Bullet-to-bullet collisions handled via exact mid-air pixel distances.
* **Procedural Systems:** Custom procedural sound engine and particle effects.

## Requirements
* Python 3.x
* Pygame

## Installation & Execution
1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install pygame
   ```
3. Run the game:
   ```bash
   python "battle_city (1) (1).py"
   ```

## Controls
* **Movement:** `Arrow Keys` or `W, A, S, D`
* **Shoot:** `SPACE` (Wait for the cooldown between shots!)
