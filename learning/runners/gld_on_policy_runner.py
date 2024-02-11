#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause

# python
import os
import time
import statistics
from collections import deque
from typing import Union

# torch
import torch
from torch.utils.tensorboard import SummaryWriter

# learning
import learning
from learning.algorithms import PPO
from learning.modules import ActorCritic, ActorCriticRecurrent
from learning.modules.discriminator import Discriminator
from learning.modules.plotter import Plotter
from learning.samplers import OfflineSampler, RandomSampler, GMMSampler, ALPGMMSampler
from learning.env import VecEnv
from learning.utils import store_code_state

import matplotlib.pyplot as plt


class FLDOnPolicyRunner:
    def __init__(self, env: VecEnv, train_cfg, log_dir=None, device="cpu"):

        self.cfg = train_cfg["runner"]
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env
        if self.env.num_privileged_obs is not None:
            num_critic_obs = self.env.num_privileged_obs
        else:
            num_critic_obs = self.env.num_obs
        actor_critic_class = eval(self.cfg["policy_class_name"])  # ActorCritic
        actor_critic: Union[ActorCritic, ActorCriticRecurrent] = actor_critic_class(
            self.env.num_obs, num_critic_obs, self.env.num_actions, **self.policy_cfg
        ).to(self.device)
        alg_class = eval(self.cfg["algorithm_class_name"])  # FLDPPO
        self.alg: PPO = alg_class(actor_critic, device=self.device, **self.alg_cfg)
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]

        self.latent_param_max, self.latent_param_min, _, _ = self.env.get_fld_latent_param_statistics()
        self.task_sampler_class = self.cfg["task_sampler_class_name"]
        self.enable_classifier = self.env.cfg.task_sampler.classifier.enabled
        if self.task_sampler_class == "OfflineSampler":
            task_sampler = OfflineSampler(self.device)
            task_sampler.load_data(self.env.cfg.fld.load_root+"/latent_params.pt")
        elif self.task_sampler_class == "RandomSampler":
            task_sampler = RandomSampler(
                self.env.fld_latent_channel * 3,
                self.latent_param_min,
                self.latent_param_max,
                device=self.device,
                curriculum_scale=self.env.cfg.task_sampler.curriculum_scale)
        elif self.task_sampler_class == "GMMSampler":
            task_sampler = GMMSampler(
                self.env.cfg.task_sampler.gmm.num_components,
                self.env.fld_latent_channel * 3,
                device=self.device,
                curriculum_scale=self.env.cfg.task_sampler.curriculum_scale,
                )
            task_sampler.load_gmm(self.env.cfg.fld.load_root+"/gmm.pt")
        elif self.task_sampler_class == "ALPGMMSampler":
            task_sampler = ALPGMMSampler(
                self.env.cfg.task_sampler.alp_gmm.init_num_components,
                self.env.cfg.task_sampler.alp_gmm.min_num_components,
                self.env.cfg.task_sampler.alp_gmm.max_num_components,
                self.env.fld_latent_channel * 3,
                self.latent_param_min,
                self.latent_param_max,
                device=self.device,
                curriculum_scale=self.env.cfg.task_sampler.curriculum_scale,
                random_type=self.env.cfg.task_sampler.alp_gmm.random_type,
                )
            if self.env.cfg.task_sampler.alp_gmm.random_type == "gmm":
                task_sampler.load_gmm(self.env.cfg.fld.load_root+"/gmm.pt")
        else:
            raise Exception(f"Unknown task sampler class {self.task_sampler_class}")
        if self.enable_classifier:
            self.task_classifier = Discriminator(self.env.fld_latent_channel * 3, self.env.cfg.task_sampler.classifier.num_classes, self.device)
            self.task_classifier.load_state_dict(torch.load(self.env.cfg.fld.load_root+"/classifier.pt")["classifier_state_dict"])
            self.task_classifier.eval()
        self.env.task_sampler = task_sampler
        self.task_sampler_update_curriculum_counter = 0
        self.task_sampler_update_counter = 0
        self.task_sampler_elite_buffer_size = self.env.cfg.task_sampler.elite_buffer_size
        self.task_sampler_library_size = self.env.cfg.task_sampler.library_size

        # init storage and model
        self.alg.init_storage(
            self.env.num_envs,
            self.num_steps_per_env,
            [self.env.num_obs],
            [self.env.num_privileged_obs],
            [self.env.num_actions],
        )

        # Log
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self.git_status_repos = [learning.__file__]

        _, _ = self.env.reset()

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        # initialize writer
        if self.log_dir is not None and self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )
        self.plotter = Plotter()
        self.fig0, self.ax0 = plt.subplots(1, 3, subplot_kw=dict(projection='polar'))
        self.fig1, self.ax1 = plt.subplots(1, 3)
        self.fig2, self.ax2 = plt.subplots(1, 4)
        self.fig3, self.ax3 = plt.subplots(1, 2)
        obs = self.env.get_observations()
        privileged_obs = self.env.get_privileged_observations()
        critic_obs = privileged_obs if privileged_obs is not None else obs
        obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)
        self.alg.actor_critic.train()  # switch to train mode (for dropout for example)

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        tot_iter = self.current_learning_iteration + num_learning_iterations
        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()
            # Rollout
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    actions = self.alg.act(obs, critic_obs)
                    obs, privileged_obs, rewards, dones, infos = self.env.step(actions)
                    critic_obs = privileged_obs if privileged_obs is not None else obs
                    obs, critic_obs, rewards, dones = (
                        obs.to(self.device),
                        critic_obs.to(self.device),
                        rewards.to(self.device),
                        dones.to(self.device),
                    )
                    self.alg.process_env_step(rewards, dones, infos)
                    
                    if self.log_dir is not None:
                        # Book keeping
                        if "episode" in infos:
                            ep_infos.append(infos["episode"])
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start

                # Learning step
                start = stop
                self.alg.compute_returns(critic_obs)

            mean_value_loss, mean_surrogate_loss = self.alg.update()

            stop = time.time()
            learn_time = stop - start

            library_size = self.env.get_library_size()
            _, performance_library = self.env.get_libraries()
            library_mean_performance = performance_library.mean()
            elite_buffer_size = self.env.get_elite_buffer_size()
            if self.task_sampler_class == "ALPGMMSampler":
                knn_buffer_size = self.env.task_sampler.get_knn_buffer_size()

            if self.log_dir is not None:
                self.log(locals())
            if it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, "model_{}.pt".format(it)))
                self.log_latent_params(it)
            
            self.check_task_sampler_curriculum_update()
            if it % self.env.cfg.task_sampler.check_update_interval == 0:
                self.check_task_sampler_update()

            ep_infos.clear()

        self.current_learning_iteration += num_learning_iterations
        self.save(os.path.join(self.log_dir, "model_{}.pt".format(self.current_learning_iteration)))
        self.log_latent_params(self.current_learning_iteration)

    def log(self, locs, width=80, pad=35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs["collection_time"] + locs["learn_time"]
        iteration_time = locs["collection_time"] + locs["learn_time"]

        ep_string = ""
        if locs["ep_infos"]:
            for key in locs["ep_infos"][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs["ep_infos"]:
                    # handle scalar and zero dimensional tensor infos
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                self.writer.add_scalar("Episode/" + key, value, locs["it"])
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        mean_std = self.alg.actor_critic.std.mean()
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs["collection_time"] + locs["learn_time"]))

        self.writer.add_scalar("Loss/value_function", locs["mean_value_loss"], locs["it"])
        self.writer.add_scalar("Loss/surrogate", locs["mean_surrogate_loss"], locs["it"])
        self.writer.add_scalar("Loss/learning_rate", self.alg.learning_rate, locs["it"])
        self.writer.add_scalar("Sampler/update_curriculum_counter", self.task_sampler_update_curriculum_counter, locs["it"])
        self.writer.add_scalar("Sampler/update_counter", self.task_sampler_update_counter, locs["it"])
        self.writer.add_scalar("Sampler/library_size", locs["library_size"], locs["it"])
        self.writer.add_scalar("Sampler/library_mean_performance", locs["library_mean_performance"], locs["it"])
        self.writer.add_scalar("Sampler/elite_buffer_size", locs["elite_buffer_size"], locs["it"])
        if self.task_sampler_class == "ALPGMMSampler":
            self.writer.add_scalar("Sampler/knn_buffer_size", locs["knn_buffer_size"], locs["it"])
        self.writer.add_scalar("Policy/mean_noise_std", mean_std.item(), locs["it"])
        self.writer.add_scalar("Perf/total_fps", fps, locs["it"])
        self.writer.add_scalar("Perf/collection time", locs["collection_time"], locs["it"])
        self.writer.add_scalar("Perf/learning_time", locs["learn_time"], locs["it"])
        if len(locs["rewbuffer"]) > 0:
            self.writer.add_scalar("Train/mean_reward", statistics.mean(locs["rewbuffer"]), locs["it"])
            self.writer.add_scalar("Train/mean_episode_length", statistics.mean(locs["lenbuffer"]), locs["it"])
            self.writer.add_scalar("Train/mean_reward/time", statistics.mean(locs["rewbuffer"]), self.tot_time)
            self.writer.add_scalar("Train/mean_episode_length/time", statistics.mean(locs["lenbuffer"]), self.tot_time)

        str = f" \033[1m Learning iteration {locs['it']}/{self.current_learning_iteration + locs['num_learning_iterations']} \033[0m "

        if len(locs["rewbuffer"]) > 0:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"""
            )
            #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
            #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")
        else:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
            )
            #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
            #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")

        log_string += ep_string
        log_string += (
            f"""{'-' * width}\n"""
            f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
            f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
            f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
            f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (
                               locs['num_learning_iterations'] - locs['it']):.1f}s\n"""
        )
        print(log_string)

    def save(self, path, infos=None):
        torch.save(
            {
                "model_state_dict": self.alg.actor_critic.state_dict(),
                "optimizer_state_dict": self.alg.optimizer.state_dict(),
                "task_sampler_state_dict": self.env.task_sampler.state_dict(),
                "iter": self.current_learning_iteration,
                "infos": infos,
            },
            path,
        )

    def check_task_sampler_curriculum_update(self):
        if self.env.task_sampler_curriculum_flag \
            and self.task_sampler_update_curriculum_counter < self.env.cfg.task_sampler.max_num_curriculum_updates \
            and self.task_sampler_update_counter < self.env.cfg.task_sampler.max_num_updates:
            self.env.task_sampler.update_curriculum()
            self.env.task_sampler_curriculum_flag = False
            self.task_sampler_update_curriculum_counter += 1

    def check_task_sampler_update(self):
        if self.task_sampler_class == "OfflineSampler":
            pass
        elif self.task_sampler_class == "RandomSampler":
            pass
        elif self.task_sampler_class == "GMMSampler":
            elite_buffer_size = self.env.get_elite_buffer_size()
            if elite_buffer_size == self.task_sampler_elite_buffer_size and self.task_sampler_update_counter < self.env.cfg.task_sampler.max_num_updates:
                elite_task_buffer, _ = self.env.get_elite_buffers()
                self.env.task_sampler.update(elite_task_buffer)
                self.env.empty_elite_buffers()
                self.env.task_sampler_curriculum_flag = False
                self.task_sampler_update_counter += 1
                self.task_sampler_update_curriculum_counter = 0
        elif self.task_sampler_class == "ALPGMMSampler":
            library_size = self.env.get_library_size()
            if library_size == self.task_sampler_library_size and self.task_sampler_update_counter < self.env.cfg.task_sampler.max_num_updates:
                task_library, performance_library = self.env.get_libraries()
                self.env.task_sampler.update(task_library, performance_library.unsqueeze(-1))
                self.env.empty_libraries()
                self.env.task_sampler_curriculum_flag = False
                self.task_sampler_update_counter += 1
                self.task_sampler_update_curriculum_counter = 0


    def log_latent_params(self, it):
        library_size = self.env.get_library_size()
        task_library, performance_library = self.env.get_libraries()
        if library_size > self.env.fld_latent_channel * 3:
            plot_pca = True
        else:
            plot_pca = False
        if plot_pca:
            self.plotter.plot_pca(self.ax2[0], [task_library], point_color=["lightgrey"], title="PCA", draw_line=False, draw_arrow=False)
            xmin, xmax = self.ax2[0].get_xlim()
            ymin, ymax = self.ax2[0].get_ylim()
            self.plotter.plot_pca_intensity(self.ax2[1], [task_library], [performance_library], cmap='YlGnBu', xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, title="Performance")
            if self.enable_classifier:
                with torch.no_grad():
                    _, task_classes = self.task_classifier(task_library).max(dim=1)
                self.plotter.plot_pca_intensity(self.ax2[3], [task_library], [task_classes], cmap='rainbow', vmin=0.0, vmax=self.env.cfg.task_sampler.classifier.num_classes-1, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, title="Class")
        self.plotter.plot_histogram(self.ax1[0], performance_library, title="Performance")
        task_library_split = task_library.split(self.env.fld_latent_channel, dim=1)
        latent_param_ymin = [ymin.min() for ymin in self.latent_param_min.split(self.env.fld_latent_channel)]
        latent_param_ymax = [ymax.max() for ymax in self.latent_param_max.split(self.env.fld_latent_channel)]
        if self.task_sampler_class == "OfflineSampler":
            mu = [None, None, None]
            var = [None, None, None]
        elif self.task_sampler_class == "RandomSampler":
            min = self.env.task_sampler.min
            max = self.env.task_sampler.max
            mu = ((min + max) / 2).unsqueeze(0).split(self.env.fld_latent_channel, dim=1)
            var = (((max - min) / 2) ** 2).unsqueeze(0).split(self.env.fld_latent_channel, dim=1)
        elif self.task_sampler_class == "GMMSampler":
            mu, var = self.env.task_sampler.gmm.get_block_parameters(self.env.fld_latent_channel)
            score_library = self.env.task_sampler.gmm.score_samples(task_library)
            self.plotter.plot_histogram(self.ax1[1], score_library, title="Score")
            self.plotter.plot_correlation(self.ax1[2], performance_library, score_library, title="Performance Score Correlation")
            if plot_pca:
                gmm_mean = self.env.task_sampler.gmm.mu
                gmm_variance = self.env.task_sampler.gmm.var
                alphas = self.env.task_sampler.gmm.pi
                self.plotter.append_pca_gmm(self.ax2[0], gmm_mean, gmm_variance, color="#ec6235", alphas=alphas)
        elif self.task_sampler_class == "ALPGMMSampler":
            if self.env.task_sampler.init_random_sampling:
                min = self.env.task_sampler.min
                max = self.env.task_sampler.max
                mu = ((min + max) / 2).unsqueeze(0).split(self.env.fld_latent_channel, dim=1)
                var = (((max - min) / 2) ** 2).unsqueeze(0).split(self.env.fld_latent_channel, dim=1)
            else:
                gmm = self.env.task_sampler.gmms.candidates[self.env.task_sampler.gmm_idx]
                mu, var = gmm.get_block_parameters(self.env.fld_latent_channel)
                if library_size > 0:
                    alp_library = self.env.task_sampler.compute_alp(task_library, performance_library.unsqueeze(-1))
                    score_library = gmm.score_samples(torch.cat((task_library, alp_library), dim=-1))
                    self.plotter.plot_histogram(self.ax1[1], score_library, title="Score")
                    self.plotter.plot_correlation(self.ax1[2], performance_library, score_library, title="Performance Score Correlation")
                if plot_pca:
                    gmm_mean = gmm.mu[:, :-1]
                    gmm_variance = gmm.var[:, :-1, :-1]
                    alphas = gmm.mu[:, -1] / gmm.mu[:, -1].sum()
                    self.plotter.append_pca_gmm(self.ax2[0], gmm_mean, gmm_variance, color="#ec6235", alphas=alphas)
                    self.plotter.plot_pca_intensity(self.ax2[2], [task_library], [alp_library], cmap='YlOrRd', xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, title="ALP")
            knn_buffer_size = self.env.task_sampler.get_knn_buffer_size()
            task_knn_buffer, performance_knn_buffer = self.env.task_sampler.get_knn_buffer()
            if plot_pca and knn_buffer_size > 0:
                self.plotter.plot_pca_intensity(self.ax3[0], [task_knn_buffer], [performance_knn_buffer], cmap='YlGnBu', title="Performance")
                if self.enable_classifier:
                    with torch.no_grad():
                        _, task_knn_classes = self.task_classifier(task_knn_buffer).max(dim=1)
                    self.plotter.plot_pca_intensity(self.ax3[1], [task_knn_buffer], [task_knn_classes], cmap='rainbow', vmin=0.0, vmax=self.env.cfg.task_sampler.classifier.num_classes-1, title="Class")
        else:
            raise Exception(f"Unknown task sampler class {self.task_sampler_class}")
        self.plotter.plot_gmm(self.ax0[0], task_library_split[0], mu[0], var[0], ymin=latent_param_ymin[0], ymax=latent_param_ymax[0], title="Frequency GMM")
        self.plotter.plot_gmm(self.ax0[1], task_library_split[1], mu[1], var[1], ymin=latent_param_ymin[1], ymax=latent_param_ymax[1], title="Amplitude GMM")
        self.plotter.plot_gmm(self.ax0[2], task_library_split[2], mu[2], var[2], ymin=latent_param_ymin[2], ymax=latent_param_ymax[2], title="Offset GMM")
        self.writer.add_figure(f"Sampler/library_spectrum", self.fig0, it)
        self.writer.add_figure(f"Sampler/library_correlation", self.fig1, it)
        self.writer.add_figure(f"Sampler/library_pca", self.fig2, it)
        self.writer.add_figure(f"Sampler/knn_pca", self.fig3, it)
        
    def load(self, path, load_optimizer=True):
        loaded_dict = torch.load(path)
        self.alg.actor_critic.load_state_dict(loaded_dict["model_state_dict"])
        self.env.task_sampler.load_state_dict(loaded_dict["task_sampler_state_dict"])
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
        self.current_learning_iteration = loaded_dict["iter"]
        return loaded_dict["infos"]

    def get_inference_policy(self, device=None):
        self.alg.actor_critic.eval()  # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_inference

    def add_git_repo_to_log(self, repo_file_path):
        self.git_status_repos.append(repo_file_path)
