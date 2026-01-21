"""
RSSM World Model - Core Architecture
Implements hierarchical Recurrent State-Space Model for healthcare
Integrates individual and system-level latent dynamics
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence
import numpy as np
from typing import Tuple, Dict, Optional

class IndividualEncoder(nn.Module):
    """
    Encodes individual patient trajectories into latent space
    Input: Patient features (age, ED visits, insurance, health status, etc.)
    Output: Latent state z_individual (32-dim)
    """
    
    def __init__(self, input_dim: int = 15, hidden_dim: int = 128, latent_dim: int = 32):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        # LSTM for temporal encoding
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )
        
        # Latent distribution parameters
        self.fc_mean = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
    def forward(self, x: torch.Tensor, hidden: Optional[Tuple] = None) -> Tuple[torch.Tensor, torch.Tensor, Tuple]:
        """
        Args:
            x: (batch, seq_len, input_dim) - Patient features over time
            hidden: Optional LSTM hidden state
            
        Returns:
            z_mean: (batch, latent_dim) - Mean of latent distribution
            z_logvar: (batch, latent_dim) - Log variance of latent distribution
            hidden: Updated LSTM hidden state
        """
        # Encode sequence
        lstm_out, hidden = self.lstm(x, hidden)
        
        # Use last timestep
        h = lstm_out[:, -1, :]  # (batch, hidden_dim)
        
        # Latent distribution
        z_mean = self.fc_mean(h)
        z_logvar = self.fc_logvar(h)
        
        return z_mean, z_logvar, hidden
    
    def sample(self, z_mean: torch.Tensor, z_logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick"""
        std = torch.exp(0.5 * z_logvar)
        eps = torch.randn_like(std)
        return z_mean + eps * std


class SystemEncoder(nn.Module):
    """
    Encodes system-level (county) dynamics into latent space
    Input: County features (capacity, providers, social determinants, etc.)
    Output: Latent state z_system (16-dim)
    """
    
    def __init__(self, input_dim: int = 10, hidden_dim: int = 64, latent_dim: int = 16):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        # GRU for temporal encoding (lighter than LSTM)
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )
        
        # Latent distribution parameters
        self.fc_mean = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
    def forward(self, x: torch.Tensor, hidden: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len, input_dim) - County features over time
            hidden: Optional GRU hidden state
            
        Returns:
            z_mean: (batch, latent_dim)
            z_logvar: (batch, latent_dim)
            hidden: Updated GRU hidden state
        """
        # Encode sequence
        gru_out, hidden = self.gru(x, hidden)
        
        # Use last timestep
        h = gru_out[:, -1, :]  # (batch, hidden_dim)
        
        # Latent distribution
        z_mean = self.fc_mean(h)
        z_logvar = self.fc_logvar(h)
        
        return z_mean, z_logvar, hidden
    
    def sample(self, z_mean: torch.Tensor, z_logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick"""
        std = torch.exp(0.5 * z_logvar)
        eps = torch.randn_like(std)
        return z_mean + eps * std


class TransitionModel(nn.Module):
    """
    Learns temporal dynamics in latent space
    Predicts next latent state given current state and action
    
    Key innovation: Bidirectional coupling between individual and system levels
    """
    
    def __init__(self, individual_latent_dim: int = 32, system_latent_dim: int = 16, 
                 action_dim: int = 5, hidden_dim: int = 128):
        super().__init__()
        
        self.individual_latent_dim = individual_latent_dim
        self.system_latent_dim = system_latent_dim
        self.action_dim = action_dim
        
        # Joint latent dimension
        self.joint_latent_dim = individual_latent_dim + system_latent_dim
        
        # Transition network
        self.transition_net = nn.Sequential(
            nn.Linear(self.joint_latent_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, self.joint_latent_dim * 2)  # mean + logvar
        )
        
    def forward(self, z_individual: torch.Tensor, z_system: torch.Tensor, 
                action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            z_individual: (batch, individual_latent_dim) - Current individual state
            z_system: (batch, system_latent_dim) - Current system state
            action: (batch, action_dim) - Action/intervention
            
        Returns:
            z_next_mean: (batch, joint_latent_dim) - Next state mean
            z_next_logvar: (batch, joint_latent_dim) - Next state log variance
        """
        # Concatenate states and action
        z_joint = torch.cat([z_individual, z_system], dim=-1)
        z_action = torch.cat([z_joint, action], dim=-1)
        
        # Predict next state distribution
        out = self.transition_net(z_action)
        z_next_mean = out[:, :self.joint_latent_dim]
        z_next_logvar = out[:, self.joint_latent_dim:]
        
        return z_next_mean, z_next_logvar
    
    def sample(self, z_mean: torch.Tensor, z_logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick"""
        std = torch.exp(0.5 * z_logvar)
        eps = torch.randn_like(std)
        return z_mean + eps * std


class IndividualDecoder(nn.Module):
    """
    Decodes individual latent state to predictions
    Predicts: ED visits, frequent ED user probability
    """
    
    def __init__(self, latent_dim: int = 32, hidden_dim: int = 64):
        super().__init__()
        
        # ED visits prediction (count)
        self.ed_visits_net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Softplus()  # Ensure positive
        )
        
        # Frequent ED user prediction (binary)
        self.frequent_ed_net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            z: (batch, latent_dim) - Individual latent state
            
        Returns:
            ed_visits_pred: (batch, 1) - Predicted ED visits
            frequent_ed_prob: (batch, 1) - Probability of frequent ED use
        """
        ed_visits_pred = self.ed_visits_net(z)
        frequent_ed_prob = self.frequent_ed_net(z)
        
        return ed_visits_pred, frequent_ed_prob


class SystemDecoder(nn.Module):
    """
    Decodes system latent state to predictions
    Predicts: Capacity breach probability, unmet demand, wait times
    
    NOVEL: These predictions are impossible with individual-only models
    """
    
    def __init__(self, latent_dim: int = 16, hidden_dim: int = 32):
        super().__init__()
        
        # Capacity breach probability
        self.capacity_breach_net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # Unmet demand (count)
        self.unmet_demand_net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Softplus()
        )
        
        # Wait time (hours)
        self.wait_time_net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Softplus()
        )
        
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            z: (batch, latent_dim) - System latent state
            
        Returns:
            capacity_breach_prob: (batch, 1)
            unmet_demand: (batch, 1)
            wait_time: (batch, 1)
        """
        capacity_breach_prob = self.capacity_breach_net(z)
        unmet_demand = self.unmet_demand_net(z)
        wait_time = self.wait_time_net(z)
        
        return capacity_breach_prob, unmet_demand, wait_time


class HealthcareRSSM(nn.Module):
    """
    Complete Hierarchical RSSM World Model for Healthcare
    
    Integrates:
    - Individual encoder (MEPS data)
    - System encoder (AHRF + CHR data)
    - Transition model (temporal dynamics)
    - Individual decoder (ED utilization)
    - System decoder (capacity constraints)
    """
    
    def __init__(self, 
                 individual_input_dim: int = 15,
                 system_input_dim: int = 10,
                 individual_latent_dim: int = 32,
                 system_latent_dim: int = 16,
                 action_dim: int = 5):
        super().__init__()
        
        # Encoders
        self.individual_encoder = IndividualEncoder(
            input_dim=individual_input_dim,
            latent_dim=individual_latent_dim
        )
        
        self.system_encoder = SystemEncoder(
            input_dim=system_input_dim,
            latent_dim=system_latent_dim
        )
        
        # Transition model
        self.transition = TransitionModel(
            individual_latent_dim=individual_latent_dim,
            system_latent_dim=system_latent_dim,
            action_dim=action_dim
        )
        
        # Decoders
        self.individual_decoder = IndividualDecoder(latent_dim=individual_latent_dim)
        self.system_decoder = SystemDecoder(latent_dim=system_latent_dim)
        
    def encode(self, individual_obs: torch.Tensor, system_obs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Encode observations to latent states"""
        # Individual encoding
        z_ind_mean, z_ind_logvar, _ = self.individual_encoder(individual_obs)
        z_individual = self.individual_encoder.sample(z_ind_mean, z_ind_logvar)
        
        # System encoding
        z_sys_mean, z_sys_logvar, _ = self.system_encoder(system_obs)
        z_system = self.system_encoder.sample(z_sys_mean, z_sys_logvar)
        
        return {
            'z_individual': z_individual,
            'z_individual_mean': z_ind_mean,
            'z_individual_logvar': z_ind_logvar,
            'z_system': z_system,
            'z_system_mean': z_sys_mean,
            'z_system_logvar': z_sys_logvar
        }
    
    def transition_step(self, z_individual: torch.Tensor, z_system: torch.Tensor, 
                       action: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Predict next latent state"""
        z_next_mean, z_next_logvar = self.transition(z_individual, z_system, action)
        z_next = self.transition.sample(z_next_mean, z_next_logvar)
        
        # Split into individual and system components
        z_next_individual = z_next[:, :self.individual_encoder.latent_dim]
        z_next_system = z_next[:, self.individual_encoder.latent_dim:]
        
        return {
            'z_next_individual': z_next_individual,
            'z_next_system': z_next_system,
            'z_next_mean': z_next_mean,
            'z_next_logvar': z_next_logvar
        }
    
    def decode(self, z_individual: torch.Tensor, z_system: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Decode latent states to predictions"""
        # Individual predictions
        ed_visits, frequent_ed_prob = self.individual_decoder(z_individual)
        
        # System predictions
        capacity_breach, unmet_demand, wait_time = self.system_decoder(z_system)
        
        return {
            'ed_visits': ed_visits,
            'frequent_ed_prob': frequent_ed_prob,
            'capacity_breach_prob': capacity_breach,
            'unmet_demand': unmet_demand,
            'wait_time': wait_time
        }
    
    def forward(self, individual_obs: torch.Tensor, system_obs: torch.Tensor, 
                action: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Full forward pass
        
        Args:
            individual_obs: (batch, seq_len, individual_input_dim)
            system_obs: (batch, seq_len, system_input_dim)
            action: (batch, action_dim)
            
        Returns:
            Dictionary with all latent states and predictions
        """
        # Encode
        encoding = self.encode(individual_obs, system_obs)
        
        # Transition
        transition = self.transition_step(
            encoding['z_individual'],
            encoding['z_system'],
            action
        )
        
        # Decode current state
        current_pred = self.decode(encoding['z_individual'], encoding['z_system'])
        
        # Decode next state
        next_pred = self.decode(transition['z_next_individual'], transition['z_next_system'])
        
        return {
            **encoding,
            **transition,
            'current_predictions': current_pred,
            'next_predictions': next_pred
        }
    
    def imagine_trajectory(self, z_individual: torch.Tensor, z_system: torch.Tensor,
                          actions: torch.Tensor, horizon: int) -> Dict[str, list]:
        """
        Imagine future trajectory through latent space
        
        NOVEL CAPABILITY: Counterfactual planning
        
        Args:
            z_individual: (batch, individual_latent_dim) - Initial individual state
            z_system: (batch, system_latent_dim) - Initial system state
            actions: (batch, horizon, action_dim) - Sequence of actions
            horizon: Number of steps to imagine
            
        Returns:
            Dictionary with trajectory of predictions
        """
        trajectory = {
            'ed_visits': [],
            'frequent_ed_prob': [],
            'capacity_breach_prob': [],
            'unmet_demand': [],
            'wait_time': []
        }
        
        z_ind = z_individual
        z_sys = z_system
        
        for t in range(horizon):
            # Decode current state
            predictions = self.decode(z_ind, z_sys)
            
            # Store predictions
            for key in trajectory:
                trajectory[key].append(predictions[key])
            
            # Transition to next state
            if t < horizon - 1:
                transition = self.transition_step(z_ind, z_sys, actions[:, t, :])
                z_ind = transition['z_next_individual']
                z_sys = transition['z_next_system']
        
        # Stack predictions
        for key in trajectory:
            trajectory[key] = torch.stack(trajectory[key], dim=1)  # (batch, horizon, 1)
        
        return trajectory


def compute_rssm_loss(model_output: Dict[str, torch.Tensor], 
                     targets: Dict[str, torch.Tensor],
                     beta: float = 1.0, gamma: float = 1.0) -> Dict[str, torch.Tensor]:
    """
    Compute RSSM training loss
    
    Components:
    1. Reconstruction loss (current state)
    2. Dynamics loss (next state prediction)
    3. KL divergence (regularization)
    
    Args:
        model_output: Dictionary from model forward pass
        targets: Dictionary with ground truth values
        beta: Weight for dynamics loss
        gamma: Weight for KL divergence
        
    Returns:
        Dictionary with loss components
    """
    # Reconstruction loss (current state)
    recon_loss = 0.0
    
    if 'ed_visits' in targets:
        recon_loss += F.mse_loss(
            model_output['current_predictions']['ed_visits'],
            targets['ed_visits']
        )
    
    if 'frequent_ed' in targets:
        recon_loss += F.binary_cross_entropy(
            model_output['current_predictions']['frequent_ed_prob'],
            targets['frequent_ed']
        )
    
    # Dynamics loss (next state prediction)
    dynamics_loss = 0.0
    
    if 'ed_visits_next' in targets:
        dynamics_loss += F.mse_loss(
            model_output['next_predictions']['ed_visits'],
            targets['ed_visits_next']
        )
    
    # KL divergence (individual)
    kl_individual = -0.5 * torch.sum(
        1 + model_output['z_individual_logvar'] 
        - model_output['z_individual_mean'].pow(2) 
        - model_output['z_individual_logvar'].exp()
    )
    
    # KL divergence (system)
    kl_system = -0.5 * torch.sum(
        1 + model_output['z_system_logvar']
        - model_output['z_system_mean'].pow(2)
        - model_output['z_system_logvar'].exp()
    )
    
    kl_loss = kl_individual + kl_system
    
    # Total loss
    total_loss = recon_loss + beta * dynamics_loss + gamma * kl_loss
    
    return {
        'total_loss': total_loss,
        'recon_loss': recon_loss,
        'dynamics_loss': dynamics_loss,
        'kl_loss': kl_loss
    }


if __name__ == "__main__":
    # Test model instantiation
    print("Testing RSSM World Model Architecture...")
    
    model = HealthcareRSSM(
        individual_input_dim=15,
        system_input_dim=10,
        individual_latent_dim=32,
        system_latent_dim=16,
        action_dim=5
    )
    
    # Test forward pass
    batch_size = 32
    seq_len = 4
    
    individual_obs = torch.randn(batch_size, seq_len, 15)
    system_obs = torch.randn(batch_size, seq_len, 10)
    action = torch.randn(batch_size, 5)
    
    output = model(individual_obs, system_obs, action)
    
    print("\n✓ Model instantiated successfully")
    print(f"  Individual latent dim: {model.individual_encoder.latent_dim}")
    print(f"  System latent dim: {model.system_encoder.latent_dim}")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test counterfactual planning
    print("\n✓ Testing counterfactual planning...")
    z_ind = output['z_individual']
    z_sys = output['z_system']
    actions = torch.randn(batch_size, 12, 5)  # 12-month horizon
    
    trajectory = model.imagine_trajectory(z_ind, z_sys, actions, horizon=12)
    
    print(f"  Imagined {trajectory['ed_visits'].shape[1]} steps ahead")
    print(f"  Trajectory shape: {trajectory['ed_visits'].shape}")
    
    print("\n✅ RSSM architecture ready for training!")
