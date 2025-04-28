import os
import random
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from collections import deque


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
        
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return (np.array(state), action, reward, np.array(next_state), done)
        
    def size(self):
        return len(self.buffer)

class QNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dims, action_dim):
        super(QNetwork, self).__init__()
        layers = []
        prev_dim = state_dim
        
        # Create hidden layers
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU()
            ])
            prev_dim = hidden_dim
            
        # Output layer
        layers.append(nn.Linear(prev_dim, action_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class DQNAgent:
    def __init__(
        self,
        env,
        hidden_dims=[128, 128],
        learning_rate=0.001,
        gamma=0.99,
        buffer_size=100000,
        batch_size=256,
        target_update=10,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        log_dir='runs/DQN_SAR',
        save_dir='saved_models_DQN'
    ):
        self.env = env
        self.device = device
        self.batch_size = batch_size
        self.gamma = gamma
        self.target_update = target_update
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.save_dir = save_dir
        
        # Initialize state and action dimensions
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n
        
        # Create networks
        self.q_network = QNetwork(self.state_dim, hidden_dims, self.action_dim).to(device)
        self.target_network = QNetwork(self.state_dim, hidden_dims, self.action_dim).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)
        
        # Initialize tensorboard writer
        self.writer = SummaryWriter(log_dir)
        
        # Create save directory if it doesn't exist
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # Tracking metrics
        self.episode_rewards = []
        self.exploration_count = 0
        self.exploitation_count = 0
        self.training_info = {
            'avg_loss': [],
            'avg_q_value': [],
            'epsilon_history': []
        }
        
    def select_action(self, state):
        if random.random() < self.epsilon:
            self.exploration_count += 1
            return random.randrange(self.action_dim)
        
        self.exploitation_count += 1
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()
    
    def train_step(self):
        if self.replay_buffer.size() < self.batch_size:
            return None
            
        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Compute current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Compute target Q values
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
            
        # Compute loss and update
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def train(self, num_episodes, minimal_size=1000, save_interval=100):
        for episode in tqdm(range(num_episodes)):
            state, _ = self.env.reset()
            episode_reward = 0
            episode_loss = []
            episode_q_values = []
            
            done = False
            while not done:
                action = self.select_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                # Store transition
                self.replay_buffer.add(state, action, reward, next_state, done)
                
                # Train if buffer has enough samples
                if self.replay_buffer.size() > minimal_size:
                    loss = self.train_step()
                    if loss is not None:
                        episode_loss.append(loss)
                        
                    # Update target network if needed
                    if self.replay_buffer.size() % self.target_update == 0:
                        self.target_network.load_state_dict(self.q_network.state_dict())
                
                state = next_state
                episode_reward += reward
                
                # Track Q-values
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                    q_values = self.q_network(state_tensor)
                    episode_q_values.append(q_values.mean().item())
            
            # Update epsilon
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
            
            # Store episode metrics
            self.episode_rewards.append(episode_reward)
            if episode_loss:
                self.training_info['avg_loss'].append(np.mean(episode_loss))
            if episode_q_values:
                self.training_info['avg_q_value'].append(np.mean(episode_q_values))
            self.training_info['epsilon_history'].append(self.epsilon)
            
            # Log to tensorboard
            self.writer.add_scalar('Episode Return', episode_reward, episode)
            self.writer.add_scalar('Training/Epsilon', self.epsilon, episode)
            if episode_loss:
                self.writer.add_scalar('Training/Average Loss', np.mean(episode_loss), episode)
            if episode_q_values:
                self.writer.add_scalar('Training/Average Q-Value', np.mean(episode_q_values), episode)
            
            # Save model periodically
            if (episode + 1) % save_interval == 0:
                self.save_model(f'dqn_model_episode_{episode + 1}.pth')
                
            # Print progress
            if (episode + 1) % 10 == 0:
                avg_reward = np.mean(self.episode_rewards[-10:])
                print(f"Episode {episode + 1}, Average Reward (last 10): {avg_reward:.2f}, Epsilon: {self.epsilon:.3f}")
        
        self.print_training_summary()
        self.writer.close()
    
    def save_model(self, filename):
        model_path = os.path.join(self.save_dir, filename)
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'training_info': self.training_info
        }, model_path)
        print(f"Model saved to {model_path}")
    
    def load_model(self, filename):
        model_path = os.path.join(self.save_dir, filename)
        checkpoint = torch.load(model_path)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.training_info = checkpoint['training_info']
        print(f"Model loaded from {model_path}")
    
    def print_training_summary(self):
        print("\nTraining Summary:")
        print(f"Total Episodes: {len(self.episode_rewards)}")
        print(f"Final Epsilon: {self.epsilon:.3f}")
        print(f"Total Exploration Actions: {self.exploration_count}")
        print(f"Total Exploitation Actions: {self.exploitation_count}")
        print(f"Final Exploration Ratio: {self.exploration_count/(self.exploration_count + self.exploitation_count):.3f}")
        print(f"Average Reward (all episodes): {np.mean(self.episode_rewards):.2f}")
        print(f"Best Episode Reward: {np.max(self.episode_rewards):.2f}")
        print(f"Worst Episode Reward: {np.min(self.episode_rewards):.2f}")



class DoubleDQNAgent:
    def __init__(
        self,
        env,
        hidden_dims=[128, 128],
        learning_rate=0.001,
        gamma=0.99,
        buffer_size=100000,
        batch_size=256,
        target_update=100,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        log_dir='runs/DDQN_SAR',
        save_dir='saved_models_DDQN'
    ):
        self.env = env
        self.device = device
        self.batch_size = batch_size
        self.gamma = gamma
        self.target_update = target_update
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.save_dir = save_dir
        
        # Initialize state and action dimensions
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n
        
        # Create networks
        self.online_network = QNetwork(self.state_dim, hidden_dims, self.action_dim).to(device)
        self.target_network = QNetwork(self.state_dim, hidden_dims, self.action_dim).to(device)
        self.target_network.load_state_dict(self.online_network.state_dict())
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(self.online_network.parameters(), lr=learning_rate)
        
        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)
        
        # Initialize tensorboard writer
        self.writer = SummaryWriter(log_dir)
        
        # Create save directory if it doesn't exist
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # Tracking metrics
        self.episode_rewards = []
        self.exploration_count = 0
        self.exploitation_count = 0
        self.training_info = {
            'avg_loss': [],
            'avg_q_value': [],
            'epsilon_history': []
        }
        
    def select_action(self, state):
        if random.random() < self.epsilon:
            self.exploration_count += 1
            return random.randrange(self.action_dim)
        
        self.exploitation_count += 1
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.online_network(state_tensor)
            return q_values.argmax().item()
    
    def train_step(self):
        if self.replay_buffer.size() < self.batch_size:
            return None
            
        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Compute current Q values
        current_q_values = self.online_network(states).gather(1, actions.unsqueeze(1))
        
        # Compute target Q values using Double DQN
        with torch.no_grad():
            # Use online network to select actions
            online_next_q_values = self.online_network(next_states)
            best_actions = online_next_q_values.argmax(1)
            
            # Use target network to evaluate those actions
            next_q_values = self.target_network(next_states)
            next_q_values = next_q_values.gather(1, best_actions.unsqueeze(1)).squeeze(1)
            
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
            
        # Compute loss and update
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def train(self, num_episodes, minimal_size=1000, save_interval=100):
        for episode in tqdm(range(num_episodes)):
            state, _ = self.env.reset()
            episode_reward = 0
            episode_loss = []
            episode_q_values = []
            
            done = False
            while not done:
                action = self.select_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                # Store transition
                self.replay_buffer.add(state, action, reward, next_state, done)
                
                # Train if buffer has enough samples
                if self.replay_buffer.size() > minimal_size:
                    loss = self.train_step()
                    if loss is not None:
                        episode_loss.append(loss)
                        
                    # Update target network if needed
                    if self.replay_buffer.size() % self.target_update == 0:
                        self.target_network.load_state_dict(self.online_network.state_dict())
                
                state = next_state
                episode_reward += reward
                
                # Track Q-values
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                    q_values = self.online_network(state_tensor)
                    episode_q_values.append(q_values.mean().item())
            
            # Update epsilon
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
            
            # Store episode metrics
            self.episode_rewards.append(episode_reward)
            if episode_loss:
                self.training_info['avg_loss'].append(np.mean(episode_loss))
            if episode_q_values:
                self.training_info['avg_q_value'].append(np.mean(episode_q_values))
            self.training_info['epsilon_history'].append(self.epsilon)
            
            # Log to tensorboard
            self.writer.add_scalar('Episode Return', episode_reward, episode)
            self.writer.add_scalar('Training/Epsilon', self.epsilon, episode)
            if episode_loss:
                self.writer.add_scalar('Training/Average Loss', np.mean(episode_loss), episode)
            if episode_q_values:
                self.writer.add_scalar('Training/Average Q-Value', np.mean(episode_q_values), episode)
            
            # Save model periodically
            if (episode + 1) % save_interval == 0:
                self.save_model(f'ddqn_model_episode_{episode + 1}.pth')
                
            # Print progress
            if (episode + 1) % 10 == 0:
                avg_reward = np.mean(self.episode_rewards[-10:])
                print(f"Episode {episode + 1}, Average Reward (last 10): {avg_reward:.2f}, Epsilon: {self.epsilon:.3f}")
        
        self.print_training_summary()
        self.writer.close()
    
    def save_model(self, filename):
        model_path = os.path.join(self.save_dir, filename)
        torch.save({
            'online_network_state_dict': self.online_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'training_info': self.training_info
        }, model_path)
        print(f"Model saved to {model_path}")
    
    def load_model(self, filename):
        model_path = os.path.join(self.save_dir, filename)
        checkpoint = torch.load(model_path)
        self.online_network.load_state_dict(checkpoint['online_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.training_info = checkpoint['training_info']
        print(f"Model loaded from {model_path}")
    
    def print_training_summary(self):
        print("\nTraining Summary:")
        print(f"Total Episodes: {len(self.episode_rewards)}")
        print(f"Final Epsilon: {self.epsilon:.3f}")
        print(f"Total Exploration Actions: {self.exploration_count}")
        print(f"Total Exploitation Actions: {self.exploitation_count}")
        print(f"Final Exploration Ratio: {self.exploration_count/(self.exploration_count + self.exploitation_count):.3f}")
        print(f"Average Reward (all episodes): {np.mean(self.episode_rewards):.2f}")
        print(f"Best Episode Reward: {np.max(self.episode_rewards):.2f}")
        print(f"Worst Episode Reward: {np.min(self.episode_rewards):.2f}")