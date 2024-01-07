"""
A module to define auxiliary diffusions needed to construct guided proposal bridge processes.
"""
import torch
from DiffusionBridge.utils import normal_logpdf


class AuxiliaryDiffusion:
    def __init__(self, model, type, initial_params):
        self.Sigma = model.Sigma
        self.T = model.T
        self.invSigma = model.invSigma
        self.time = model.time
        self.stepsizes = model.stepsizes
        self.f = model["f"]
        self.type = type
        self.params = {
            name: torch.tensor(value, requires_grad=True)
            for name, value in initial_params.items()
        }

    def auxiliary_f(self, t, x):
        if self.type == "brownian":
            return self.params["alpha"]

        if self.type == "ou":
            return self.params["alpha"] + self.params["beta"] * x

    def transition_mean(self, t, x):
        if self.type == "brownian":
            return x + t * self.params["alpha"]

        if self.type == "ou":
            ratio = self.params["alpha"] / self.params["beta"]
            return ratio + (x - ratio) * torch.exp(-self.params["beta"] * t)

    def transition_var(self, t):
        if self.type == "brownian":
            return t * self.Sigma

        if self.type == "ou":
            return (1.0 - torch.exp(-2.0 * self.params["beta"] * t)) / (
                2.0 * self.params["beta"]
            )

    def log_transition(self, s, xs, t, xt):
        return normal_logpdf(
            xt, self.transition_mean(t - s, xs), self.transition_var(t - s)
        )

    def grad_logh(self, terminal_state, t, x):
        if self.type == "brownian":
            return (
                self.invSigma * (terminal_state - x) / (self.T - t)
                - self.invSigma * self.params["alphas"]
            )

        if self.type == "ou":
            return (
                (terminal_state - self.transition_mean(self.T - t, x))
                * torch.exp(-self.params["alpha"] * (self.T - t))
                / self.transition_var(self.T - t)
            )

    def log_radon_nikodym(self, trajectories):
        N = trajectories.shape[0]
        M = trajectories.shape[1] - 1
        initial_state = trajectories[:, 0, :]
        terminal_state = trajectories[:, -1, :]
        G = torch.zeros(N, M)

        for m in range(M):
            t_current = self.time[m]
            X_current = trajectories[:, m, :]
            diff_f = self.f(t_current, X_current) - self.auxiliary_f(
                t_current, X_current
            )  # size (N, d)
            grad_logh = self.grad_logh(
                terminal_state, t_current, X_current
            )  # size (N, d)
            G[:, m] = torch.sum(diff_f * grad_logh, 1) * self.stepsizes[m]  # size (N)

        log_rn = torch.sum(G, 1) + self.log_transition(
            0.0, initial_state, self.T, terminal_state
        )  # size (N)

        return log_rn