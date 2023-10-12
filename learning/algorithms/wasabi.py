#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause

# torch
import torch
import torch.nn as nn
import torch.optim as optim

# learning
from learning.modules import ActorCritic
from learning.modules.discriminator import Discriminator
from learning.storage import RolloutStorage
from learning.storage.replay_buffer import ReplayBuffer


class WASABI:
    actor_critic: ActorCritic
    discriminator: Discriminator

    def __init__(
        self,
        actor_critic,
        discriminator,
        wasabi_expert_data,
        wasabi_state_normalizer,
        wasabi_style_reward_normalizer,
        num_learning_epochs=1,
        num_mini_batches=1,
        clip_param=0.2,
        gamma=0.998,
        lam=0.95,
        value_loss_coef=1.0,
        entropy_coef=0.0,
        policy_learning_rate=1e-3,
        max_grad_norm=1.0,
        use_clipped_value_loss=True,
        schedule="fixed",
        desired_kl=0.01,
        device="cpu",
        discriminator_learning_rate=0.000025,
        discriminator_momentum=0.9,
        discriminator_weight_decay=0.0005,
        discriminator_gradient_penalty_coef=5,
        discriminator_loss_function="MSELoss",
        discriminator_num_mini_batches=10,
        wasabi_replay_buffer_size=100000,
        **kwargs,
    ):
        if kwargs:
            print("ActorCritic.__init__ got unexpected arguments, which will be ignored: "
                  + str([key for key in kwargs.keys()]))

        self.device = device

        self.desired_kl = desired_kl
        self.schedule = schedule

        # PPO components
        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        self.storage = None  # initialized later
        self.transition = RolloutStorage.Transition()  # actor_critic transition

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.policy_learning_rate = policy_learning_rate

        self.policy_optimizer = optim.Adam(self.actor_critic.parameters(), lr=self.policy_learning_rate)

        # Discriminator components
        self.discriminator = discriminator
        self.discriminator.to(self.device)
        self.wasabi_policy_data = ReplayBuffer(discriminator.observation_dim, discriminator.observation_horizon, wasabi_replay_buffer_size, device)
        self.wasabi_expert_data = wasabi_expert_data
        self.wasabi_state_normalizer = wasabi_state_normalizer
        self.wasabi_style_reward_normalizer = wasabi_style_reward_normalizer

        # Discriminator parameters
        self.discriminator_learning_rate = discriminator_learning_rate
        self.discriminator_momentum = discriminator_momentum
        self.discriminator_weight_decay = discriminator_weight_decay
        self.discriminator_gradient_penalty_coef = discriminator_gradient_penalty_coef
        self.discriminator_loss_function = discriminator_loss_function
        self.discriminator_num_mini_batches = discriminator_num_mini_batches

        if self.discriminator_loss_function == "WassersteinLoss":
            discriminator_optimizer = optim.RMSprop
        else:
            discriminator_optimizer = optim.SGD
        self.discriminator_optimizer = discriminator_optimizer(
                                                    self.discriminator.parameters(),
                                                    lr=self.discriminator_learning_rate,
                                                    momentum=self.discriminator_momentum,
                                                    weight_decay=self.discriminator_weight_decay,
                                                )

    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape):
        self.storage = RolloutStorage(
            num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape, self.device
        )

    def test_mode(self):
        self.actor_critic.test()

    def train_mode(self):
        self.actor_critic.train()

    def act(self, obs, critic_obs, wasabi_observation_buf):
        if self.actor_critic.is_recurrent:
            self.transition.hidden_states = self.actor_critic.get_hidden_states()
        # Compute the actions and values
        self.transition.actions = self.actor_critic.act(obs).detach()
        self.transition.values = self.actor_critic.evaluate(critic_obs).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        # need to record obs and critic_obs before env.step()
        self.transition.observations = obs
        self.transition.critic_observations = critic_obs
        self.wasabi_observation_buf = wasabi_observation_buf.clone()
        return self.transition.actions

    def process_env_step(self, rewards, dones, infos, wasabi_obs):
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        # Bootstrapping on time outs
        if "time_outs" in infos:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * infos["time_outs"].unsqueeze(1).to(self.device), 1
            )

        # Record the transition
        self.storage.add_transitions(self.transition)
        wasabi_observation_buf = torch.cat((self.wasabi_observation_buf[:, 1:], wasabi_obs.unsqueeze(1)), dim=1)
        self.wasabi_policy_data.insert(wasabi_observation_buf)
        self.transition.clear()
        self.actor_critic.reset(dones)

    def compute_returns(self, last_critic_obs):
        last_values = self.actor_critic.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    def update(self):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_wasabi_loss = 0
        mean_grad_pen_loss = 0
        mean_policy_pred = 0
        mean_expert_pred = 0
        
        # Policy update
        if self.actor_critic.is_recurrent:
            generator = self.storage.reccurent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for (
            obs_batch,
            critic_obs_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            hid_states_batch,
            masks_batch,
        ) in generator:

            self.actor_critic.act(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
            value_batch = self.actor_critic.evaluate(
                critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1]
            )
            mu_batch = self.actor_critic.action_mean
            sigma_batch = self.actor_critic.action_std
            entropy_batch = self.actor_critic.entropy

            # KL
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                        + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        axis=-1,
                    )
                    kl_mean = torch.mean(kl)

                    if kl_mean > self.desired_kl * 2.0:
                        self.policy_learning_rate = max(1e-5, self.policy_learning_rate / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.policy_learning_rate = min(1e-2, self.policy_learning_rate * 1.5)

                    for param_group in self.policy_optimizer.param_groups:
                        param_group["lr"] = self.policy_learning_rate

            # Surrogate loss
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value function loss
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            ppo_loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

            # Gradient step
            self.policy_optimizer.zero_grad()
            ppo_loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.policy_optimizer.step()
            
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()

        # Discriminator update
        wasabi_policy_generator = self.wasabi_policy_data.feed_forward_generator(
            self.discriminator_num_mini_batches,
            self.storage.num_envs * self.storage.num_transitions_per_env // self.discriminator_num_mini_batches)
        wasabi_expert_generator = self.wasabi_expert_data.feed_forward_generator(
            self.discriminator_num_mini_batches,
            self.storage.num_envs * self.storage.num_transitions_per_env // self.discriminator_num_mini_batches)

        for sample_wasabi_policy, sample_wasabi_expert in zip(wasabi_policy_generator, wasabi_expert_generator):

            # Discriminator loss
            policy_state_buf = torch.zeros_like(sample_wasabi_policy)
            expert_state_buf = torch.zeros_like(sample_wasabi_expert)
            if self.wasabi_state_normalizer is not None:
                for i in range(self.discriminator.observation_horizon):
                    with torch.no_grad():
                        policy_state_buf[:, i] = self.wasabi_state_normalizer.normalize(sample_wasabi_policy[:, i])
                        expert_state_buf[:, i] = self.wasabi_state_normalizer.normalize(sample_wasabi_expert[:, i])
            policy_d = self.discriminator(policy_state_buf.flatten(1, 2))
            expert_d = self.discriminator(expert_state_buf.flatten(1, 2))
            if self.discriminator_loss_function == "BCEWithLogitsLoss":
                expert_loss = torch.nn.BCEWithLogitsLoss()(expert_d, torch.ones_like(expert_d))
                policy_loss = torch.nn.BCEWithLogitsLoss()(policy_d, torch.zeros_like(policy_d))
            elif self.discriminator_loss_function == "MSELoss":
                expert_loss = torch.nn.MSELoss()(expert_d, torch.ones(expert_d.size(), device=self.device))
                policy_loss = torch.nn.MSELoss()(policy_d, -1 * torch.ones(policy_d.size(), device=self.device))
            elif self.discriminator_loss_function == "WassersteinLoss":
                expert_loss = -expert_d.mean()
                policy_loss = policy_d.mean()
            else:
                raise ValueError("Unexpected loss function specified")
            wasabi_loss = 0.5 * (expert_loss + policy_loss)
            grad_pen_loss = self.discriminator.compute_grad_pen(sample_wasabi_expert,
                                                                lambda_=self.discriminator_gradient_penalty_coef)

            # Gradient step
            discriminator_loss = wasabi_loss + grad_pen_loss
            self.discriminator_optimizer.zero_grad()
            discriminator_loss.backward()
            self.discriminator_optimizer.step()

            if self.wasabi_state_normalizer is not None:
                self.wasabi_state_normalizer.update(sample_wasabi_policy[:, 0])
                self.wasabi_state_normalizer.update(sample_wasabi_expert[:, 0])

            mean_wasabi_loss += wasabi_loss.item()
            mean_grad_pen_loss += grad_pen_loss.item()
            mean_policy_pred += policy_d.mean().item()
            mean_expert_pred += expert_d.mean().item()

        policy_num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= policy_num_updates
        mean_surrogate_loss /= policy_num_updates

        discriminator_num_updates = self.discriminator_num_mini_batches
        mean_wasabi_loss /= discriminator_num_updates
        mean_grad_pen_loss /= discriminator_num_updates
        mean_policy_pred /= discriminator_num_updates
        mean_expert_pred /= discriminator_num_updates

        self.storage.clear()

        return mean_value_loss, mean_surrogate_loss, mean_wasabi_loss, mean_grad_pen_loss, mean_policy_pred, mean_expert_pred
