from humanoid_gym import LEGGED_GYM_ROOT_DIR
from isaacgym.torch_utils import (
    quat_rotate,
)
from learning.utils import get_base_quat_from_base_ang_vel
from learning.modules.fld import FLD
from learning.modules.plotter import Plotter
from learning.modules.gmm import GaussianMixture
import os
import torch
import matplotlib.pyplot as plt


class FLDEvaluate:
    
    def __init__(self, state_idx_dict, observation_horizon, num_consecutives, device):
        self.state_idx_dict = state_idx_dict
        self.observation_horizon = observation_horizon
        self.num_consecutives = num_consecutives
        self.dim_of_interest = torch.cat(
            [
                torch.tensor(ids, device=device, dtype=torch.long, requires_grad=False)
                for state, ids in state_idx_dict.items()
                if ((state != "base_pos") and (state != "base_quat"))
                ]
            )
        self.observation_dim = self.dim_of_interest.size(0)
        self.device = device
        self.dt = 0.02
        self.observation_start_dim = 7

        self.fld = FLD(self.observation_dim, observation_horizon, latent_dim, device, encoder_shape=[64, 64], decoder_shape=[64, 64])
        runs = os.listdir(log_dir_root)
        runs.sort()
        last_run = os.path.join(log_dir_root, runs[-1])
        self.load_run = last_run
        models = [file for file in os.listdir(self.load_run) if "model" in file]
        models.sort(key=lambda m: "{0:0>15}".format(m))
        model = models[-1]
        loaded_dict = torch.load(os.path.join(self.load_run, model))
        self.fld.load_state_dict(loaded_dict["fld_state_dict"])
        statistics_dict = torch.load(os.path.join(self.load_run, "statistics.pt"))
        self.state_transitions_mean, self.state_transitions_std = statistics_dict["state_transitions_mean"], statistics_dict["state_transitions_std"]

        self.plotter = Plotter()
        fig1, self.ax1 = plt.subplots(4, 1)
        fig2, self.ax2 = plt.subplots(8, 5)
        fig3, self.ax3 = plt.subplots()
        fig4, self.ax4 = plt.subplots(1, 3, subplot_kw=dict(projection='polar'))


    def prepare_data(self):
        datasets_root = os.path.join(LEGGED_GYM_ROOT_DIR + "/resources/robots/mit_humanoid/datasets/misc")
        motion_data = os.listdir(datasets_root)
        motion_name_set = [data.replace('motion_data_', '').replace('.pt', '') for data in motion_data if "combined" not in data and ".pt" in data]
        motion_data_collection = []

        for i, motion_name in enumerate(motion_name_set):
            motion_path = os.path.join(datasets_root, "motion_data_" + motion_name + ".pt")
            motion_data = torch.load(motion_path, map_location=self.device)[:, :, self.dim_of_interest]
            loaded_num_trajs, loaded_num_steps, loaded_obs_dim = motion_data.size()
            print(f"[Motion Loader] Loaded motion {motion_name} with {loaded_num_trajs} trajectories, {loaded_num_steps} steps with {loaded_obs_dim} dimensions.")
            motion_data_collection.append(motion_data.unsqueeze(0))
            # compute latent dynamics error
            state_transitions = (motion_data - self.state_transitions_mean) / self.state_transitions_std
            error = self.fld.get_dynamics_error(state_transitions, k=4).mean().item()
            print(f"[Motion Loader] Motion {motion_name} dynamics error: {error}")

        motion_data_collection = torch.cat(motion_data_collection, dim=0)
        motion_data_collection = motion_data_collection.unfold(2, self.observation_horizon + self.num_consecutives - 1, 1).swapaxes(-2, -1)
        self.state_transitions_data = (motion_data_collection - self.state_transitions_mean) / self.state_transitions_std

    def evaluate(self):
        self.fld.eval()
        self.num_motions = self.state_transitions_data.size(0)
        self.eval_manifold_collection = []
        self.fit_gmm()
        self.sample_latent()

    def fit_gmm(self):
        covariance_type = "full"
        num_components = 8
        gmm = GaussianMixture(num_components, latent_dim * 3, device=device, covariance_type=covariance_type)

        with torch.no_grad():
            for i in range(self.num_motions):
                eval_traj = self.state_transitions_data[i, 0, :, :self.observation_horizon, :].swapaxes(1, 2)
                pred_dynamics, latent, signal, params = self.fld(eval_traj)
                pred = pred_dynamics[0]

                phase = params[0]
                amplitude = params[2]
                manifold = torch.hstack(
                    (
                        amplitude * torch.sin(2.0 * torch.pi * phase),
                        amplitude * torch.cos(2.0 * torch.pi * phase),
                        )
                    )
                self.eval_manifold_collection.append(manifold.cpu())
            
            # fit GMM
            print(f"[FLD Evaluate] Fitting GMM...")
            all_state_transitions = self.state_transitions_data[:, :, :, :self.observation_horizon, :].flatten(0, 2).swapaxes(1, 2)
            _, _, _, all_params = self.fld(all_state_transitions)
            all_frequency = all_params[1]
            all_amplitude = all_params[2]
            all_offset = all_params[3]
            latent_params = torch.cat((all_frequency, all_amplitude, all_offset), dim=1)
            gmm.fit(latent_params)
            mu, var = gmm.get_block_parameters(latent_dim)
            self.plotter.plot_gmm(self.ax4[0], all_frequency.view(self.num_motions, -1, latent_dim), mu[0], var[0], title="Frequency GMM")
            self.plotter.plot_gmm(self.ax4[1], all_amplitude.view(self.num_motions, -1, latent_dim), mu[1], var[1], title="Amplitude GMM")
            self.plotter.plot_gmm(self.ax4[2], all_offset.view(self.num_motions, -1, latent_dim), mu[2], var[2], title="Offset GMM")

            torch.save(
                latent_params,
                self.load_run + f"/latent_params.pt"
                )
            torch.save(
                {
                    "gmm_state_dict": gmm.state_dict(),
                    },
                self.load_run + f"/gmm.pt"
                )

    def sample_latent(self):
        with torch.no_grad():
            eval_traj = self.state_transitions_data[0, 0, :, :self.observation_horizon, :].swapaxes(1, 2)
            pred_dynamics, latent, signal, params = self.fld(eval_traj)
            latent_sample_frequency = params[1][0, :] * torch.ones_like(params[1], device=device, dtype=torch.float, requires_grad=False)
            latent_sample_amplitude = params[2][0, :] * torch.ones_like(params[2], device=device, dtype=torch.float, requires_grad=False)
            latent_sample_offset = params[3][0, :] * torch.ones_like(params[3], device=device, dtype=torch.float, requires_grad=False)
            latent_sample_phase = params[0][0, :] + latent_sample_frequency * torch.arange(eval_traj.size(0), device=device, dtype=torch.float, requires_grad=False).unsqueeze(-1) * self.dt

            latent_sample_z = latent_sample_amplitude.unsqueeze(-1) * torch.sin(2 * torch.pi * (latent_sample_frequency.unsqueeze(-1) * self.fld.args + latent_sample_phase.unsqueeze(-1))) + latent_sample_offset.unsqueeze(-1)
            latent_sample_manifold = torch.hstack(
                (
                    latent_sample_amplitude * torch.sin(2.0 * torch.pi * latent_sample_phase),
                    latent_sample_amplitude * torch.cos(2.0 * torch.pi * latent_sample_phase),
                    )
                )
            
            self.eval_manifold_collection.append(latent_sample_manifold.cpu())

            decoded_traj_pred = self.fld.decoder(latent_sample_z)
            decoded_traj_raw = decoded_traj_pred.swapaxes(1, 2)
            decoded_traj = decoded_traj_raw * self.state_transitions_std + self.state_transitions_mean
            decoded_traj = torch.cat(
                (
                    decoded_traj[0, :, :],
                    decoded_traj[1:, -1, :],
                ),
                dim=0
            ).unsqueeze(0)

            plot_traj_index = 0
            self.plotter.plot_circles(self.ax1[0], latent_sample_phase[plot_traj_index], latent_sample_amplitude[plot_traj_index], title="Learned Phase Timing"  + " " + str(latent_dim) + "x" + str(2), show_axes=False)
            self.plotter.plot_curves(self.ax1[1], latent_sample_z[plot_traj_index], -1.0, 1.0, -2.0, 2.0, title="Latent Parametrized Signal" + " " + str(latent_dim) + "x" + str(observation_horizon), show_axes=False)
            self.plotter.plot_curves(self.ax1[2], decoded_traj_pred[plot_traj_index], -1.0, 1.0, -5.0, 5.0, title="Curve Reconstruction" + " " + str(self.observation_dim) + "x" + str(observation_horizon), show_axes=False)
            self.plotter.plot_curves(self.ax1[3], decoded_traj_pred[plot_traj_index].flatten(0, 1).unsqueeze(0), -1.0, 1.0, -5.0, 5.0, title="Curve Reconstruction (Flattened)" + " " + str(1) + "x" + str(self.fld.input_channel*observation_horizon), show_axes=False)

            for j in range(latent_dim):
                phase = latent_sample_phase[:, j]
                frequency = latent_sample_frequency[:, j]
                amplitude = latent_sample_amplitude[:, j]
                offset = latent_sample_offset[:, j]
                self.plotter.plot_phase_1d(self.ax2[j, 0], phase, amplitude, title=("1D Phase Values" if j==0 else None), show_axes=False)
                self.plotter.plot_phase_2d(self.ax2[j, 1], phase, amplitude, title=("2D Phase Vectors" if j==0 else None), show_axes=False)
                self.plotter.plot_curves(self.ax2[j, 2], frequency.unsqueeze(0), -1.0, 1.0, 0.0, 4.0, title=("Frequencies" if j==0 else None), show_axes=False)
                self.plotter.plot_curves(self.ax2[j, 3], amplitude.unsqueeze(0), -1.0, 1.0, 0.0, 1.0, title=("Amplitudes" if j==0 else None), show_axes=False)
                self.plotter.plot_curves(self.ax2[j, 4], offset.unsqueeze(0), -1.0, 1.0, -1.0, 1.0, title=("Offsets" if j==0 else None), show_axes=False)

            self.plotter.plot_pca(self.ax3, self.eval_manifold_collection, "Phase Manifold (" + str(self.num_motions) + " Random Sequences)")

        decoded_root = os.path.join(LEGGED_GYM_ROOT_DIR + "/resources/robots/mit_humanoid/datasets/decoded")
        if not os.path.exists(decoded_root):
            os.makedirs(decoded_root)
        decoded_path = os.path.join(decoded_root, "motion_data.pt")

        # fill global pos and quat from integrated lin_vel and ang_vel
        decoded_traj_buf = torch.zeros(decoded_traj.size(0), decoded_traj.size(1), 52, device=device, dtype=torch.float, requires_grad=False)
        decoded_base_ang_vel = decoded_traj[:, :, torch.tensor(state_idx_dict["base_ang_vel"]) - self.observation_start_dim]
        init_base_quat = torch.tensor([0.0, 0.0, 0.0, 1.0], device=device, dtype=torch.float, requires_grad=False).repeat(decoded_traj.size(0), 1)
        # integrate base ang vel to get base quat
        decoded_base_quat = get_base_quat_from_base_ang_vel(decoded_base_ang_vel, self.dt, source_frame="local", init_base_quat=init_base_quat)

        init_base_pos = torch.tensor([0.0, 0.0, 0.66], device=device, dtype=torch.float, requires_grad=False)
        decoded_base_lin_vel = decoded_traj[:, :, torch.tensor(state_idx_dict["base_lin_vel"]) - self.observation_start_dim]
        decoded_base_lin_vel_global = quat_rotate(decoded_base_quat.flatten(0, 1), decoded_base_lin_vel.flatten(0, 1)).view(-1, decoded_traj.size(1), 3)
        # integrate base lin vel to get base pos
        decoded_base_pos_change = torch.cumsum(decoded_base_lin_vel_global * self.dt, dim=1)
        decoded_base_pos = decoded_base_pos_change + init_base_pos

        decoded_traj_buf[:, :, torch.tensor(state_idx_dict["base_pos"])] = decoded_base_pos
        decoded_traj_buf[:, :, torch.tensor(state_idx_dict["base_quat"])] = decoded_base_quat

        decoded_traj_buf[:, :, self.dim_of_interest] = decoded_traj

        torch.save(decoded_traj_buf, decoded_path)

        self.ax3.legend()


if __name__ == "__main__":
    state_idx_dict = {
        "base_pos": [0, 1, 2],
        "base_quat": [3, 4, 5, 6],
        "base_lin_vel": [7, 8, 9],
        "base_ang_vel": [10, 11, 12],
        "projected_gravity": [13, 14, 15],
        "dof_pos_leg_l": [16, 17, 18, 19, 20],
        "dof_pos_arm_l": [21, 22, 23, 24],
        "dof_pos_leg_r": [25, 26, 27, 28, 29],
        "dof_pos_arm_r": [30, 31, 32, 33],
    }
    observation_horizon = 51
    latent_dim = 8
    num_consecutives = 50
    device = "cuda"
    log_dir_root = LEGGED_GYM_ROOT_DIR + "/logs/flat_mit_humanoid/fld/"
    fld_evaluate = FLDEvaluate(state_idx_dict, observation_horizon, num_consecutives, device)
    fld_evaluate.prepare_data()
    fld_evaluate.evaluate()
    plt.show()
