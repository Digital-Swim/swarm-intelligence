# Implementing PSO

import numpy as np
from entities import Agent

def pso(num_iterations, env):
 
    # Initialize global best position and score
    global_best_position = None
    global_best_score = float('-inf')
    
    for iteration in range(num_iterations):
        for agent in env.agents:
            # Evaluate the agent's score based on its current position
            agent.score = env.get_score_at_position(agent.position)
            
            # Update the agent's personal best position and score
            if agent.score > agent.best_position_score:
                agent.best_position = agent.position.copy()
                agent.best_position_score = agent.score
            
            # Update the global best position and score
            if agent.score > global_best_score:
                global_best_position = agent.position.copy()
                global_best_score = agent.score
        
        # Update each agent's velocity and position based on PSO equations
        for agent in env.agents:
            inertia_weight = 0.5  # Inertia weight
            cognitive_weight = 1.5  # Cognitive weight
            social_weight = 1.5  # Social weight
            
            r1, r2 = np.random.rand(2)  # Random factors
            
            # Update velocity
            cognitive_component = cognitive_weight * r1 * (agent.best_position - agent.position)
            social_component = social_weight * r2 * (global_best_position - agent.position)
            agent.velocity = inertia_weight * agent.velocity + cognitive_component + social_component
            
            # Update position
            agent.position += agent.velocity
            
            # Ensure the position is within bounds (0, 1)
            agent.position = np.clip(agent.position, 0, 1)
        
        # Optionally, visualize the agents' positions at each iteration
        env.visualize_agents(env.agents, iteration)
    
    return global_best_position, global_best_score