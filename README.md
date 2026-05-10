# Battle City AI 🎮

A Python and Pygame-based recreation of the classic arcade game *Battle City*, featuring advanced Artificial Intelligence.

This project was developed as a collaborative effort by **Nimrah Shahid, Saman, and Saweba**.

Unlike the original game which relied on simple randomized logic, this project serves as an AI sandbox where different tank classes are driven by distinct, academic AI architectures ranging from Simple Reflex Agents to Adversarial Minimax algorithms.

---

## 🚀 Contributors
- **Nimrah Shahid** – Core AI logic, game engine, architecture design  
- **Saman** – AI behavior design, debugging, testing support  
- **Saweba** – Game mechanics, environment design, feature support  

---

## 🧠 Features

### Advanced Enemy AI
The game features 5 distinct enemy classes, each implementing a unique AI architecture:

- **Basic Tank (Simple Reflex Agent):**  
  Uses IF-THEN rules and **Breadth-First Search (BFS)** pathfinding. Dynamically clears destructible walls in its path.

- **Fast Tank (Goal-Based Agent):**  
  Uses **Greedy Best-First Search** to aggressively target the player’s Eagle. Can get stuck in local minima, demonstrating limitations of greedy strategies.

- **Armor Tank (Model-Based Agent):**  
  Uses **A\* Pathfinding** with internal state tracking (HP). On critical damage, switches to defensive mode and uses BFS to find steel walls for cover.

- **Power Tank (Utility-Based Agent):**  
  Computes real-time utility scores to decide between attacking the player or rushing the Eagle using **A\*** navigation.

- **Boss Tank (Adversarial Agent):**  
  Multi-phase boss using **Minimax Algorithm with Alpha-Beta Pruning**, simulating future game states up to 4 turns ahead.

---

## 🗺️ Dynamic Environment

- 🧱 **Destructible Terrain:** Brick walls can be destroyed, changing AI pathfinding in real time  
- 🌲 **Forest Zones:** Tanks can hide and become invisible in forests  
- 💥 **Pixel-Perfect Physics:** Accurate collision detection system  
- 🔊 **Procedural Systems:** Custom sound effects and particle generation  

---

## 📦 Requirements

- Python 3.x  
- :contentReference[oaicite:0]{index=0}  

---

## ⚙️ Installation & Execution

### 1. Clone the repository
```bash
git clone <your-repo-link>
cd battle-city-ai
2. Install dependencies
pip install pygame
3. Run the game
python "battle_city (1) (1).py"

## 🎮 Controls

**Move:** Arrow Keys / W A S D
**Shoot:** SPACE (with cooldown timing)

## 📌 Notes

Designed as an AI learning project for pathfinding and adversarial search algorithms
Focuses on applying academic AI concepts in real-time gameplay
Optimized for educational demonstration of multiple AI strategies

## ⭐ If you like this project

Feel free to star the repository and explore the AI logic!
