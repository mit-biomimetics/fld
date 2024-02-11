import torch
from sklearn.decomposition import PCA
from matplotlib.patches import Ellipse
import matplotlib.colors as mcolors
from matplotlib.colors import to_rgba

class Plotter:
    def __init__(self) -> None:
        self.pca = PCA(n_components=2)

    def plot_pca(self, ax, manifold_collection, title=None, point_color=None, draw_line=True, draw_arrow=True):
        ax.cla()

        point_alpha = 0.3
        line_alpha = 0.2
        arrow_alpha = 1.0
        arrow_step = 50
        arrow_size = 0.015
        arrow_power = 1.0
        arrow_color = (0.25, 0.25, 0.5)

        num_steps = [len(manifold) for manifold in manifold_collection]
        manifolds = torch.cat(manifold_collection, dim=0).cpu()
        
        manifolds_pca = torch.tensor(self.pca.fit_transform(manifolds)).split(num_steps, dim=0)

        for i, manifold in enumerate(manifolds_pca):
            if draw_line:
                ax.plot(manifold[:, 0], manifold[:, 1], color=(0.0, 0.0, 0.0), alpha=line_alpha)
            if point_color is None:
                ax.scatter(manifold[:, 0], manifold[:, 1], alpha=point_alpha, label=i)
            else:
                ax.scatter(manifold[:, 0], manifold[:, 1], color=point_color[i], alpha=point_alpha, label=i)
            if draw_arrow:
                for j in range(0, len(manifold) - arrow_step, arrow_step):
                    d = torch.norm(manifold[j, :], dim=-1)
                    d = torch.pow(d, arrow_power)
                    ax.arrow(
                        manifold[j, 0],
                        manifold[j, 1],
                        manifold[j + 1, 0] - manifold[j, 0],
                        manifold[j + 1, 1] - manifold[j, 1],
                        alpha=arrow_alpha, width=d * arrow_size, color=arrow_color
                        )
        ax.legend()
        ax.set_axis_off()
        if title != None:
            ax.set_title(title)

    def append_pca_gmm(self, ax, mean, variance, color=None, alphas=None):
        mean = mean.cpu()
        variance = variance.cpu()
        n_components = mean.size(0)
        if color is None:
            color = 'red'
        if alphas is None:
            alphas = [0.8] * n_components
        else:
            alphas = alphas.cpu().tolist()
        for i in range(n_components):
            mu = mean[i]
            var = variance[i]
            mu_transformed = torch.tensor(self.pca.transform(mu.unsqueeze(0)), dtype=torch.float).squeeze(0)
            eigval, eigvec = torch.linalg.eigh(var)
            pca_components = torch.tensor(self.pca.components_, dtype=torch.float)
            eigvec_transformed = torch.matmul(pca_components, eigvec)
            var_projected = torch.matmul(eigvec_transformed, torch.matmul(torch.diag(eigval), eigvec_transformed.T))
            eigval_transformed, eigvec_transformed = torch.linalg.eigh(var_projected)
            width = eigval_transformed[0].sqrt() * 2
            height = eigval_transformed[1].sqrt() * 2
            std_scale = 3.0
            angle = torch.atan2(eigvec_transformed[1, 0], eigvec_transformed[0, 0]) * 180 / torch.pi
            ell = Ellipse(mu_transformed, width * std_scale, height * std_scale, angle=angle, fc=to_rgba(color, 0.2 * alphas[i]), ec=to_rgba(color, alphas[i]), lw=3)
            ax.add_artist(ell)
    
    def plot_pca_intensity(self, ax, manifold_collection, values, cmap='YlOrRd', vmin=0.0, vmax=1.0, xmin=None, xmax=None, ymin=None, ymax=None, title=None):
        ax.cla()

        point_alpha = 0.5

        num_steps = [len(manifold) for manifold in manifold_collection]
        manifolds = torch.cat(manifold_collection, dim=0).cpu()
        
        manifolds_pca = torch.tensor(self.pca.transform(manifolds)).split(num_steps, dim=0)

        for i, manifold in enumerate(manifolds_pca):
            ax.scatter(manifold[:, 0], manifold[:, 1], alpha=point_alpha, c=values[i].cpu(), cmap=cmap, vmin=vmin, vmax=vmax)
    
        ax.set_axis_off()
        if title != None:
            ax.set_title(title)
        if xmin != None and xmax != None:
            ax.set_xlim(xmin, xmax)
        if ymin != None and ymax != None:
            ax.set_ylim(ymin, ymax)

    def plot_distribution(self, ax, values, title):
        ax.cla()
        values = values.cpu()
        means = torch.mean(values, dim=0)
        std = torch.std(values, dim=0)
        args = torch.arange(values.size(1))
        ax.bar(args, means, yerr=std, align='center', alpha=0.5, ecolor='black', capsize=10)
        ax.set_xlabel('Channel')
        ax.set_xticks(args)
        ax.set_title(title)
        ax.yaxis.grid(True)

    def plot_histogram(self, ax, values, title):
        ax.cla()
        values = values.cpu()
        ax.hist(values, bins=50, density=True)
        ax.set_title(title)
        ax.yaxis.grid(True)                

    def plot_gmm(self, ax, data, pred_mean, pred_var, color=None, ymin=None, ymax=None, title=None):
        # ax.cla()
        data = data.cpu()
        if pred_mean is not None and pred_var is not None:
            pred_mean = pred_mean.cpu()
            pred_var = torch.linalg.cholesky(pred_var).diagonal(dim1=-2, dim2=-1) if pred_var.dim() == 3 else pred_var
            pred_std = torch.sqrt(pred_var.cpu())
        if data.dim() == 2:
            if color == None:
                color = "lightgrey"
            args = torch.arange(data.size(1))
            theta_args = args * torch.pi / 4
            ax.plot(theta_args.repeat(data.size(0), 1).cpu(), data, "o", color="lightgrey", alpha=0.01, markersize=10, mew=0.0)
        elif data.dim() == 3:
            if color == None:
                color = list(mcolors.TABLEAU_COLORS.keys())
            args = torch.arange(data.size(2))
            theta_args = args * torch.pi / 4
            for i in range(data.size(0)):
                ax.plot(theta_args.repeat(data.size(1), 1).cpu(), data[i], "o", color=color[i], alpha=0.01, markersize=10, mew=0.0)
        if pred_mean is not None and pred_var is not None:
            theta_args_mean = torch.cat([theta_args, theta_args[0].unsqueeze(0)], dim=0)
            pred_mean = torch.cat([pred_mean, pred_mean[:, 0].unsqueeze(1)], dim=1)
            pred_std = torch.cat([pred_std, pred_std[:, 0].unsqueeze(1)], dim=1)
            for i in range(pred_mean.size(0)):
                ax.errorbar(theta_args_mean, pred_mean[i], yerr=pred_std[i], fmt="o-", markersize=4, capsize=5, alpha=0.5, linewidth=3)
        ax.set_xlabel('Channel')
        ax.set_xticks(theta_args)
        ax.set_xticklabels(args.tolist())
        if title != None:
            ax.set_title(title)
        if ymin != None and ymax != None:
            ymin = ymin.cpu()
            ymax = ymax.cpu()
            ax.set_ylim(ymin, ymax)
        ax.yaxis.grid(True)

    def plot_correlation(self, ax, performance, score, title=None):
        ax.cla()
        performance = performance.cpu()
        score = score.cpu()
        ax.plot(performance, score, "o", color="tab:grey", alpha=0.002, markersize=10)
        ax.set_xlabel('Performance')
        ax.set_ylabel('Score')
        ax.set_xlim(0.0, 1.0)
        if title != None:
            ax.set_title(title)

    def plot_circles(self, ax, phase, amplitude, title=None, show_axes=True):
        ax.cla()
        phase = phase.cpu()
        amplitude = amplitude.cpu()

        aspect = 0.5
        ax.set_aspect(aspect)
        channel = phase.shape[0]
        ax.set_xlim(0.0, channel + 1.0)
        ax.set_ylim(-1.0, 1.0)
        theta = torch.linspace(0.0, 2 * torch.pi, 100)

        for i in range(channel):
            p = phase[i]
            a = amplitude[i]
            x1 = aspect * a * torch.cos(theta) + i + 1
            x2 = a * torch.sin(theta)
            ax.plot(x1, x2)
            line_x1 = [i + 1, i + 1 + aspect * a * torch.cos(2 * torch.pi * p)]
            line_x2 = [0.0,  a * torch.sin(2 * torch.pi * p)]
            ax.plot(line_x1, line_x2, color=(0, 0, 0))

        if title != None:
            ax.set_title(title)
        if show_axes == False:
            ax.axes.xaxis.set_visible(False)
            ax.axes.yaxis.set_visible(False)

    def plot_curves(self, ax, values, xmin, xmax, ymin, ymax, title=None, show_axes=True):
        ax.cla()
        values = values.cpu()
        args = torch.linspace(xmin, xmax, values.size(1)).repeat(values.size(0), 1)
        ax.plot(args.swapaxes(0, 1), values.swapaxes(0, 1))
        ax.set_ylim(ymin, ymax)
        if title != None:
            ax.set_title(title)
        if show_axes == False:
            ax.axes.xaxis.set_visible(False)
            ax.axes.yaxis.set_visible(False)

    def plot_phase_1d(self, ax, phase, amplitude, title=None, show_axes=True):
        ax.cla()
        phase = phase.cpu()
        amplitude = amplitude.cpu()
        
        phase = torch.where(phase < 0, phase, phase + 1)
        phase = phase % 1.0
        args = torch.arange(phase.size(0))
        amplitude = torch.clip(amplitude, 0.0, 1.0)
        for i in range(1, phase.size(0)):
            ax.plot(
                [args[i - 1].item(), args[i].item()],
                [phase[i - 1].item(), phase[i].item()],
                color=(0.0, 0.0, 0.0),
                alpha=amplitude[i].item()
                )
        ax.set_ylim(0.0, 1.0)

        if title != None:
            ax.set_title(title)
        if show_axes == False:
            ax.axes.xaxis.set_visible(False)
            ax.axes.yaxis.set_visible(False)

    def plot_phase_2d(self, ax, phase, amplitude, title=None, show_axes=True):
        ax.cla()
        phase = phase.cpu()
        amplitude = amplitude.cpu()

        args = torch.arange(phase.size(0))

        phase_x1 = amplitude * torch.sin(2 * torch.pi * phase)
        phase_x2 = amplitude * torch.cos(2 * torch.pi * phase)

        ax.plot(args, phase_x1)
        ax.plot(args, phase_x2)
        ax.set_ylim(-1.0, 1.0)
        
        if title != None:
            ax.set_title(title)
        if show_axes == False:
            ax.axes.xaxis.set_visible(False)
            ax.axes.yaxis.set_visible(False)