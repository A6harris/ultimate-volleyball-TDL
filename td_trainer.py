from mlagents.trainers.trainer import Trainer
from mlagents.trainers.policy import Policy
from mlagents.trainers.policy.torch_policy import TorchPolicy
from mlagents.trainers.torch.networks import SimpleNetwork
from mlagents.trainers.brain import BrainParameters
from mlagents.trainers.behavior_id_utils import BehaviorIdentifiers
from mlagents.trainers.settings import TrainerSettings, NetworkSettings
from mlagents.trainers.trajectory import Trajectory
from mlagents.trainers.buffer import Buffer
from mlagents.trainers.agent_processor import AgentManagerQueue
from mlagents.trainers.stats import StatsReporter, StatsSummary
from mlagents_envs.base_env import BehaviorSpec
from mlagents.torch_utils import torch, default_device

import numpy as np
import os
from typing import Dict, List, Optional, Callable, Tuple, Any, cast


class TDNetwork(SimpleNetwork):
    """
    Q-Network for TD Learning that is compatible with ML-Agents
    """
    def __init__(
        self,
        observation_shapes: List[Tuple[int, ...]],
        network_settings: NetworkSettings,
        action_spec: BehaviorSpec.ActionSpec,
        stream_names: List[str],
        conditional_sigma: bool = False,
        deterministic: bool = False,
    ):
        super().__init__(
            observation_shapes,
            network_settings,
            action_spec,
            stream_names,
            conditional_sigma,
            deterministic,
        )
        num_outputs = sum(self.action_spec.discrete_branches)
        self.q_head = torch.nn.Linear(network_settings.hidden_units, num_outputs)
        
    def forward(self, inputs: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        encoding = self._encoder(inputs)
        q_values = self.q_head(encoding)
        
        # Split Q-values by action branches
        branch_q_values = []
        start_idx = 0
        for branch_size in self.action_spec.discrete_branches:
            branch_q_values.append(q_values[:, start_idx:start_idx + branch_size])
            start_idx += branch_size
        
        output = {}
        for name, branch_q in zip(self.stream_names, branch_q_values):
            output[name] = branch_q
            
        return output


class TDPolicy(TorchPolicy):
    """
    Policy implementation for TD Learning
    """
    def __init__(
        self,
        brain_parameters: BrainParameters,
        trainer_settings: TrainerSettings,
        network_settings: NetworkSettings,
        model_path: str,
        seed: int,
        behavior_spec: BehaviorSpec,
    ):
        # Initialize the parent TorchPolicy
        super().__init__(
            brain_parameters,
            trainer_settings,
            network_settings,
            model_path,
            seed,
            behavior_spec,
        )
        
        # Setup for epsilon-greedy exploration
        self.epsilon = trainer_settings.hyperparameters.epsilon
        self.epsilon_decay = trainer_settings.hyperparameters.epsilon_decay
        self.epsilon_min = trainer_settings.hyperparameters.epsilon_min
        
    def create_torch_network(self) -> TDNetwork:
        """
        Creates the TD network for this policy
        """
        return TDNetwork(
            self.behavior_spec.observation_shapes,
            self.network_settings,
            self.behavior_spec.action_spec,
            self.stream_names,
        )
    
    def evaluate(self, decision_requests):
        """
        Evaluate the policy for the agent observations in decision_requests
        and with the epsilon-greedy behavior for exploration
        """
        with torch.no_grad():
            self.network.eval()
            inputs = self._get_policy_eval_inputs(decision_requests)
            q_values = self.network(inputs)
            
            # epsilon-greedy action selection
            actions = {}
            for name, branch_q in q_values.items():
                if np.random.random() < self.epsilon:
                    # Choose random action
                    branch_size = branch_q.shape[1]
                    actions[name] = torch.randint(0, branch_size, (branch_q.shape[0],))
                else:
                    # Choose greedy action
                    actions[name] = torch.argmax(branch_q, dim=1)
            
            # Decay epsilon
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            
            return {"action": actions}
    
    def update_network(self, batch):
        """
        Update the network using TD learning
        """
        self.network.train()
        self.optimizer.zero_grad()
        
        value_estimates, _ = self.network(batch["observations"])
        
        # Get the current Q-values
        current_q_values = []
        for name in self.stream_names:
            q_vals = value_estimates[name]
            actions = batch["actions"][name]
            current_q = torch.gather(q_vals, 1, actions.unsqueeze(-1)).squeeze(-1)
            current_q_values.append(current_q)
        
        # Calculate target Q-values
        with torch.no_grad():
            target_value_estimates, _ = self.target_network(batch["next_observations"])
            
            # Get max Q-values for each action branch
            max_q_values = []
            for name in self.stream_names:
                max_q = torch.max(target_value_estimates[name], dim=1)[0]
                max_q_values.append(max_q)
            
            # Combine max Q-values from different branches
            max_q_combined = torch.mean(torch.stack(max_q_values), dim=0)
            
            # Calculate target Q = r + gamma * max_a Q'(s',a')
            target_q = batch["rewards"] + self.gamma * (1.0 - batch["done"]) * max_q_combined
        
        # Calculate TD loss for each branch
        losses = []
        for current_q in current_q_values:
            # MSE loss between current Q and target Q
            losses.append(torch.nn.functional.mse_loss(current_q, target_q))
        
        # Total loss is the average of the branch losses
        loss = torch.mean(torch.stack(losses))
        
        # Backpropagate
        loss.backward()
        
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.trainer_settings.hyperparameters.grad_clip)
        
        # Update weights
        self.optimizer.step()
        
        # Update target network
        if self.steps % self.target_update_interval == 0:
            self.update_target_network()
        
        self.steps += 1
        
        update_stats = {
            "Loss": loss.item(),
            "Epsilon": self.epsilon,
        }
        
        return update_stats


class TDTrainer(Trainer):
    """
    Temporal Difference (TD) Learning trainer for Unity ML-Agents.
    """
    def __init__(
        self,
        brain_name: str,
        trainer_settings: TrainerSettings,
        training: bool,
        load: bool,
        seed: int,
        artifact_path: str,
    ):
        """
        Initialize the TD Trainer
        """
        super().__init__(
            brain_name, trainer_settings, training, load, seed, artifact_path
        )
        
        # Initialize the replay buffer
        self.buffer_size = self.trainer_settings.hyperparameters.buffer_size
        self.batch_size = self.trainer_settings.hyperparameters.batch_size
        self.buffer = Buffer()
        
        # Counter for buffer updates
        self.update_buffer_step = 0
        
        # Steps between policy updates
        self.steps_per_update = self.trainer_settings.hyperparameters.steps_per_update
        
        # Create the statistics reporter
        self.stats_reporter = StatsReporter(brain_name)
        
    def _process_trajectory(self, trajectory: Trajectory) -> None:
        """
        Process a trajectory and add it to the replay buffer
        """
        super()._process_trajectory(trajectory)
        
        # Add trajectory samples to the buffer
        if trajectory.steps:
            buffer_trajectory = trajectory.to_sample_batches(trajectory.all_group_spec)
            for agent_id in buffer_trajectory:
                self.buffer[agent_id].extend(buffer_trajectory[agent_id])
        
        # Check if it's time to update
        self.update_buffer_step += 1
        if (
            self.update_buffer_step >= self.steps_per_update
            and len(self.buffer) >= self.batch_size
            and self.is_ready_update
        ):
            self._update_policy()
            self.update_buffer_step = 0
        
    def _update_policy(self) -> None:
        """
        Update the policy based on experiences in the buffer
        """
        # Sample from the buffer
        batch_size = min(self.batch_size, len(self.buffer))
        sampled_batch = self.buffer.sample_batch(batch_size)
        
        # Update the policy with the sampled batch
        update_stats = self.policy.update_network(sampled_batch)
        
        # Update the stats reporter
        for stat_name, stat_value in update_stats.items():
            self.stats_reporter.add_stat(stat_name, stat_value)
    
    def save_model(self) -> None:
        """
        Save the model
        """
        model_checkpoint = os.path.join(
            self.artifact_path, f"{self.brain_name}_{self.get_step}.pt"
        )
        self.policy.save(model_checkpoint)
        
    def create_policy(
        self,
        parsed_behavior_id: BehaviorIdentifiers,
        behavior_spec: BehaviorSpec,
        create_graph: bool = False,
    ) -> TDPolicy:
        """
        Create a TDPolicy
        """
        brain_parameters = BrainParameters(
            parsed_behavior_id.brain_name,
            behavior_spec.observation_shapes,
            behavior_spec.action_spec,
            parsed_behavior_id.brain_name,
        )
        
        return TDPolicy(
            brain_parameters,
            self.trainer_settings,
            self.trainer_settings.network_settings,
            self.artifact_path,
            self.seed,
            behavior_spec,
        )