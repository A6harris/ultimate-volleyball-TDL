"""
Standalone TD Learning implementation for Ultimate Volleyball.
This doesn't depend on the internal ML-Agents trainer architecture.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.base_env import ActionTuple
import random
from collections import deque
import os
import argparse
import time
import yaml
from datetime import datetime
import tensorboard
from torch.utils.tensorboard import SummaryWriter

# Check for CUDA
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""
    
    def __init__(self, buffer_size, batch_size):
        """Initialize a ReplayBuffer object.
        
        Parameters:
        -----------
        buffer_size : int
            maximum size of buffer
        batch_size : int
            size of each training batch
        """
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = (state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=min(len(self.memory), self.batch_size))
        
        # Convert to tensors
        states = torch.from_numpy(np.vstack([e[0] for e in experiences])).float().to(device)
        actions = torch.from_numpy(np.vstack([e[1] for e in experiences])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e[2] for e in experiences])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e[3] for e in experiences])).float().to(device)
        dones = torch.from_numpy(np.vstack([e[4] for e in experiences]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, dones)
    
    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

class QNetwork(nn.Module):
    """Actor (Policy) Model."""
    
    def __init__(self, state_size, action_branches, hidden_size=256):
        """Initialize parameters and build model.
        
        Parameters:
        -----------
        state_size : int
            Dimension of each state
        action_branches : list
            List with the size of each action branch
        hidden_size : int
            Number of nodes in hidden layers
        """
        super(QNetwork, self).__init__()
        
        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        
        # Separate output layers for each action branch
        self.action_heads = nn.ModuleList([
            nn.Linear(hidden_size, branch_size) for branch_size in action_branches
        ])
        
    def forward(self, state):
        """Build a network that maps state -> action values."""
        shared_features = self.shared(state)
        
        # Get Q-values for each action branch
        branch_q_values = [head(shared_features) for head in self.action_heads]
        
        return branch_q_values

class TDAgent:
    """Interacts with and learns from the environment using TD learning."""
    
    def __init__(self, state_size, action_branches, config, seed=0):
        """Initialize a TD Learning Agent.
        
        Parameters:
        -----------
        state_size : int
            dimension of each state
        action_branches : list
            list with the size of each action branch
        config : dict
            configuration dictionary with hyperparameters
        seed : int
            random seed
        """
        self.state_size = state_size
        self.action_branches = action_branches
        self.seed = random.seed(seed)
        
        # Hyperparameters
        self.batch_size = config.get('batch_size', 128)
        self.buffer_size = config.get('buffer_size', 20480)
        self.gamma = config.get('gamma', 0.96)
        self.learning_rate = config.get('learning_rate', 0.0002)
        self.tau = config.get('tau', 0.001)  # For soft update of target parameters
        self.update_every = config.get('update_every', 4)
        self.epsilon = config.get('epsilon_start', 1.0)
        self.epsilon_decay = config.get('epsilon_decay', 0.995)
        self.epsilon_min = config.get('epsilon_min', 0.1)
        
        # Q-Networks (local and target)
        self.qnetwork_local = QNetwork(state_size, action_branches, 
                                       config.get('hidden_size', 256)).to(device)
        self.qnetwork_target = QNetwork(state_size, action_branches, 
                                        config.get('hidden_size', 256)).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.learning_rate)
        
        # Replay memory
        self.memory = ReplayBuffer(self.buffer_size, self.batch_size)
        
        # Initialize time step (for updating every update_every steps)
        self.t_step = 0
        
    def step(self, state, action, reward, next_state, done):
        """Save experience in replay memory, and learn if it's time to update."""
        # Save experience in replay memory
        action_array = np.array(action, dtype=np.int32).reshape(1, -1)
        self.memory.add(state, action_array, np.array([[reward]]), next_state, np.array([[done]]))
        
        # Learn every update_every time steps
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0 and len(self.memory) > self.batch_size:
            experiences = self.memory.sample()
            self.learn(experiences)
            
    def act(self, state, training=True):
        """Returns actions for given state as per current policy.
        
        Parameters:
        -----------
        state : array_like
            current state
        training : bool
            whether to use epsilon-greedy action selection
        
        Returns:
        --------
        list
            List of actions for each branch
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        
        self.qnetwork_local.eval()
        with torch.no_grad():
            branch_q_values = self.qnetwork_local(state)
        self.qnetwork_local.train()
        
        # Epsilon-greedy action selection
        actions = []
        for i, q_values in enumerate(branch_q_values):
            if training and random.random() < self.epsilon:
                actions.append(random.randint(0, self.action_branches[i] - 1))
            else:
                actions.append(q_values.cpu().data.numpy().argmax())
                
        # Decay epsilon
        if training:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
                
        return actions
    
    def learn(self, experiences):
        """Update value parameters using given batch of experience tuples.
        
        Parameters:
        -----------
        experiences : tuple
            (states, actions, rewards, next_states, dones)
        """
        states, actions, rewards, next_states, dones = experiences
        
        # Get max predicted Q values for each branch
        self.qnetwork_target.eval()
        with torch.no_grad():
            next_branch_q_values = self.qnetwork_target(next_states)
            max_next_q_values = torch.stack([q_values.max(1)[0] for q_values in next_branch_q_values])
            # Average across branches
            max_next_q = max_next_q_values.mean(0).unsqueeze(1)
            # Compute Q targets for current states
            q_targets = rewards + (self.gamma * max_next_q * (1 - dones))
        
        # Get current Q values
        branch_q_values = self.qnetwork_local(states)
        
        # Compute loss for each branch
        losses = []
        for i, q_values in enumerate(branch_q_values):
            branch_actions = actions[:, i].unsqueeze(1)
            branch_q = q_values.gather(1, branch_actions)
            losses.append(nn.functional.mse_loss(branch_q, q_targets))
        
        # Sum losses and perform optimization
        loss = sum(losses)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Soft update target network
        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)
        
        return loss.item()
    
    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
            
    def save(self, filename):
        """Save the model weights."""
        torch.save(self.qnetwork_local.state_dict(), filename)
        
    def load(self, filename):
        """Load model weights."""
        self.qnetwork_local.load_state_dict(torch.load(filename, map_location=device))
        self.qnetwork_target.load_state_dict(torch.load(filename, map_location=device))

def train_volleyball_td(
    env_path,
    config_path='config/Volleyball_TD.yaml',
    run_id=None,
    output_dir='results',
    num_episodes=10000,
    max_steps=1000,
    time_scale=1.0
):
    """
    Train agents using TD Learning.
    
    Parameters:
    -----------
    env_path : str
        Path to the Unity environment executable
    config_path : str
        Path to the configuration YAML file
    run_id : str
        Identifier for this training run
    output_dir : str
        Directory to save results
    num_episodes : int
        Number of episodes to train
    max_steps : int
        Maximum steps per episode
    time_scale : float
        Time scale factor for Unity simulation (1.0 = real-time)
    """
    # Load configuration
    with open(config_path, 'r') as f:
        try:
            config_data = yaml.safe_load(f)
            hyperparams = config_data['behaviors']['Volleyball']['hyperparameters']
            network_settings = config_data['behaviors']['Volleyball']['network_settings']
        except (yaml.YAMLError, KeyError) as e:
            print(f"Error loading config: {e}")
            print("Using default hyperparameters")
            hyperparams = {
                'batch_size': 128,
                'buffer_size': 20480,
                'learning_rate': 0.0002,
                'gamma': 0.96,
                'epsilon_start': 1.0,
                'epsilon_decay': 0.995,
                'epsilon_min': 0.1,
                'update_every': 4,
                'tau': 0.001
            }
            network_settings = {
                'hidden_size': 256
            }
    
    # Combine all settings
    config = {**hyperparams, **network_settings}
    
    # Create run ID if not provided
    if run_id is None:
        run_id = f"volleyball_td_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Create output directory
    run_dir = os.path.join(output_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)
    
    # Create TensorBoard writer
    writer = SummaryWriter(log_dir=run_dir)
    
    # Configure the environment
    engine_configuration_channel = EngineConfigurationChannel()
    env = UnityEnvironment(
        file_name=env_path,
        seed=0,
        side_channels=[engine_configuration_channel],
        no_graphics=True
    )
    engine_configuration_channel.set_configuration_parameters(time_scale=time_scale)
    
    # Reset the environment
    env.reset()
    
    # Get the behavior name and spec
    behavior_name = list(env.behavior_specs.keys())[0]
    spec = env.behavior_specs[behavior_name]
    
    # Get observation and action specs
    observation_size = spec.observation_shapes[0][0]
    action_spec = spec.action_spec
    action_branches = action_spec.discrete_branches
    
    print(f"Observation size: {observation_size}")
    print(f"Action branches: {action_branches}")
    
    # Create TD agents
    blue_agent = TDAgent(observation_size, action_branches, config, seed=0)
    purple_agent = TDAgent(observation_size, action_branches, config, seed=1)
    
    # Training loop
    total_steps = 0
    for episode in range(1, num_episodes+1):
        env.reset()
        
        decision_steps, terminal_steps = env.get_steps(behavior_name)
        blue_score = 0
        purple_score = 0
        episode_steps = 0
        episode_start_time = time.time()
        
        while episode_steps < max_steps and len(terminal_steps) == 0:
            # Get current observations
            if 0 in decision_steps and 1 in decision_steps:
                blue_obs = decision_steps[0].obs[0]
                purple_obs = decision_steps[1].obs[0]
                
                # Select actions
                blue_actions = blue_agent.act(blue_obs)
                purple_actions = purple_agent.act(purple_obs)
                
                # Combine actions
                all_actions = np.array([blue_actions, purple_actions])
                
                # Create ActionTuple
                action_tuple = ActionTuple()
                action_tuple.add_discrete(all_actions)
                
                # Execute actions
                env.set_actions(behavior_name, action_tuple)
                env.step()
                
                # Get new state and rewards
                new_decision_steps, new_terminal_steps = env.get_steps(behavior_name)
                
                # Process terminal steps
                if len(new_terminal_steps) > 0:
                    for agent_id in new_terminal_steps:
                        if agent_id == 0:  # Blue agent
                            reward = new_terminal_steps[agent_id].reward
                            blue_score += reward
                            blue_agent.step(
                                blue_obs, 
                                blue_actions, 
                                reward, 
                                new_terminal_steps[agent_id].obs[0], 
                                True
                            )
                        elif agent_id == 1:  # Purple agent
                            reward = new_terminal_steps[agent_id].reward
                            purple_score += reward
                            purple_agent.step(
                                purple_obs, 
                                purple_actions, 
                                reward, 
                                new_terminal_steps[agent_id].obs[0], 
                                True
                            )
                    break
                else:
                    # Process decision steps
                    for agent_id in new_decision_steps:
                        if agent_id == 0:  # Blue agent
                            reward = new_decision_steps[agent_id].reward
                            blue_score += reward
                            blue_agent.step(
                                blue_obs, 
                                blue_actions, 
                                reward, 
                                new_decision_steps[agent_id].obs[0], 
                                False
                            )
                        elif agent_id == 1:  # Purple agent
                            reward = new_decision_steps[agent_id].reward
                            purple_score += reward
                            purple_agent.step(
                                purple_obs, 
                                purple_actions, 
                                reward, 
                                new_decision_steps[agent_id].obs[0], 
                                False
                            )
                    
                    # Update decision steps
                    decision_steps = new_decision_steps
            else:
                # If we don't have observations for both agents, step the environment
                env.step()
                decision_steps, terminal_steps = env.get_steps(behavior_name)
            
            episode_steps += 1
            total_steps += 1
        
        # Episode finished - log results
        episode_duration = time.time() - episode_start_time
        steps_per_second = episode_steps / episode_duration
        
        # Log to TensorBoard
        writer.add_scalar('Volleyball/Episode Length', episode_steps, total_steps)
        writer.add_scalar('Volleyball/Blue Agent/Cumulative Reward', blue_score, total_steps)
        writer.add_scalar('Volleyball/Purple Agent/Cumulative Reward', purple_score, total_steps)
        writer.add_scalar('Volleyball/Blue Agent/Epsilon', blue_agent.epsilon, total_steps)
        writer.add_scalar('Volleyball/Purple Agent/Epsilon', purple_agent.epsilon, total_steps)
        writer.add_scalar('Volleyball/Steps Per Second', steps_per_second, total_steps)
        
        # Print progress
        print(f"Episode {episode}/{num_episodes} - Steps: {episode_steps} - Blue: {blue_score:.2f} - Purple: {purple_score:.2f} - Epsilon: {blue_agent.epsilon:.2f} - SPS: {steps_per_second:.1f}")
        
        # Save model periodically
        if episode % 100 == 0:
            blue_agent.save(os.path.join(run_dir, f"blue_agent_ep{episode}.pth"))
            purple_agent.save(os.path.join(run_dir, f"purple_agent_ep{episode}.pth"))
    
    # Save final models
    blue_agent.save(os.path.join(run_dir, "blue_agent_final.pth"))
    purple_agent.save(os.path.join(run_dir, "purple_agent_final.pth"))
    
    # Close environment and TensorBoard writer
    env.close()
    writer.close()
    
    print(f"Training complete! Models saved to {run_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Volleyball TD Agents')
    parser.add_argument('--env-path', type=str, required=True, 
                        help='Path to Unity environment executable')
    parser.add_argument('--config', type=str, default='config/Volleyball_TD.yaml',
                        help='Path to configuration file')
    parser.add_argument('--run-id', type=str, default=None,
                        help='Identifier for this training run')
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Directory to save results')
    parser.add_argument('--episodes', type=int, default=10000,
                        help='Number of episodes to train')
    parser.add_argument('--max-steps', type=int, default=1000,
                        help='Maximum steps per episode')
    parser.add_argument('--time-scale', type=float, default=1.0,
                        help='Time scale for Unity simulation')
    
    args = parser.parse_args()
    
    train_volleyball_td(
        env_path=args.env_path,
        config_path=args.config,
        run_id=args.run_id,
        output_dir=args.output_dir,
        num_episodes=args.episodes,
        max_steps=args.max_steps,
        time_scale=args.time_scale
    )