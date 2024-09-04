# Solving the Snake Game: Algorithms vs. Reinforcement Learning

This project explores different approaches to solving the classic Snake game by comparing traditional algorithms with Reinforcement Learning (RL) methods.

## Overview

The goal is to investigate how different strategies perform in solving the Snake game. The project is divided into two main sections:

1. **Algorithms**: This section covers the use of traditional algorithms to play the game. Various algorithms are implemented and tested to understand their efficiency and effectiveness in navigating the snake through the grid.

2. **Reinforcement Learning (RL)**: In this section, a machine learning approach is applied to train an agent to play the Snake game using Reinforcement Learning techniques. The focus is on understanding how the RL agent learns over time and how it compares with the algorithmic approaches.

## Algorithmic Approach

- **If-Else Strategy**: 
  - The snake always aims directly for the apple.
  - Simple decision-making with basic if-else logic.
  - Limited effectiveness as it can lead to frequent collisions and getting trapped.

- **A* Algorithm**:
  - The snake finds the shortest path to the apple while avoiding collisions.
  - Dynamically updates the path as obstacles change.
  - More effective than the if-else strategy but can still struggle in complex scenarios.

- **Hamiltonian Cycle**:
  - The snake follows a precomputed safe path that covers the entire grid.
  - Occasionally skips parts of the cycle to reach the apple faster.
  - Provides the safest path but may be less efficient in terms of time.

Detailed presentation: https://www.canva.com/design/DAFiMtnlggc/KR4pgUoE82h7TEn9ZqPKnA/edit?utm_content=DAFiMtnlggc&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton

## Reinforcement Learning Approach (branch AI)

- **Model Architecture**: 
  - A fully connected neural network with 4 layers.
  - The network learns to predict the best actions based on the current state of the game.

- **Training Process**:
  - Initialize a Q-Network with random weights.
  - Initialize a target Q-Network with the same structure and random weights.
  - Create a buffer memory to store experiences: (states, actions, rewards, next states, done flags).

  **Training Loop**:
  - While training:
    - For 2000 steps:
      - Choose the best action based on the Q-Network with a probability of E (where E decays over time).
      - Store the current state in the buffer memory.
      
    - Every C steps:
      - Sample a batch of experiences from the memory.
      - Compute the expected Q value for the current state.
      - Update the Q value using the reward and the target Q-Network.
      - Compute the loss as the difference between the new and old Q values.
      - Perform a backward pass and update the Q-Network weights using gradient descent.
      - Soft update the target Q-Network weights: `target_w = (1-t) * target_w + t * w`.

Detailed presentation: https://www.canva.com/design/DAFiMtifH0Q/Q0f96S651hhHU2fIMZ8ASQ/edit?utm_content=DAFiMtifH0Q&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton

## Results and Comparison

- **Algorithms vs. RL**:
  - A detailed comparison of the results obtained from the algorithmic approaches and the RL agent. This includes metrics such as average score, completion time, and learning efficiency.
  - The machine learning approach, while not as effective as the algorithmic methods, demonstrates some strategic behavior. Unlike the algorithms, which follow predictable patterns, the RL agent adapts and shows a degree of decision-making, even if it's not always optimal.

## Conclusion

This project highlights the potential of Reinforcement Learning in solving complex problems like the Snake game, even though traditional algorithms may still outperform it in this context. The RL agent, despite its limitations, shows a capacity for strategic play, which could be further refined with more advanced techniques or longer training times.

## Future Work

- Explore ways to improve the RL agent's performance, such as by tuning hyperparameters or experimenting with different network architectures.
- Apply the insights gained from this project to other games or problem-solving scenarios.
- Investigate hybrid approaches that combine algorithms and RL for potentially better results. Instead of letting the snake training on itself, using its own data, we could incorporate some data getting from the algorithm phrase
- Try on Genetic Algorithm
