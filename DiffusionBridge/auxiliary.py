"""
A module to define auxiliary diffusions needed to construct guided proposal bridge processes.
"""

import torch
from DiffusionBridge.utils import normal_logpdf, mv_normal_logpdf


class AuxiliaryDiffusion:

    def __init__(self, model, auxiliary_type, initial_params, requires_grad=True):
        self.Sigma = model.Sigma
        self.T = model.T
        self.invSigma = model.invSigma
        self.time = model.time
        self.stepsizes = model.stepsizes
        self.f = model.f
        self.type = auxiliary_type
        self.params = {
            name: torch.tensor(value, requires_grad=requires_grad)
            for name, value in initial_params.items()
        }

    def project_params(self):
        if self.type == "ou-full":
            U, _, V = torch.svd(self.params["eigvecs"])
            self.params["eigvecs"] = U @ V.T  # for orthogonality

    def return_params(self):
        return {name: tensor.detach().clone() for name, tensor in self.params.items()}

    def auxiliary_f(self, t, x):
        if self.type == "bm":
            return self.params["alpha"]

        if self.type == "ou":
            return self.params["alpha"] - self.params["beta"] * x

        if self.type == "ou-full":
            mat = (
                self.params["eigvecs"]
                @ torch.diag(self.params["eigvals"])
                @ self.params["eigvecs"].T
            )
            return (self.params["theta"] - x) @ mat

    def transition_mean(self, t, x):
        if self.type == "bm":
            return x + t * self.params["alpha"]

        if self.type == "ou":
            ratio = self.params["alpha"] / self.params["beta"]
            return ratio + (x - ratio) * torch.exp(-self.params["beta"] * t)

        if self.type == "ou-full":
            mat_exp = (
                self.params["eigvecs"]
                @ torch.diag(torch.exp(-t * self.params["eigvals"]))
                @ self.params["eigvecs"].T
            )
            return (x - self.params["theta"]) @ mat_exp + self.params["theta"]

    def transition_var(self, t):
        if self.type == "bm":
            return t * self.Sigma

        if self.type == "ou":
            return (
                self.Sigma
                * (1.0 - torch.exp(-2.0 * self.params["beta"] * t))
                / (2.0 * self.params["beta"])
            )

        if self.type == "ou-full":
            cov_eigvals = (
                0.5
                * self.Sigma
                * (1.0 / self.params["eigvals"])
                * (1.0 - torch.exp(-2.0 * t * self.params["eigvals"]))
            )  # return eigenvalues in this case
            return cov_eigvals

    def log_transition(self, s, xs, t, xt):
        if self.type == "ou-full":
            cov_eigvals = self.transition_var(t - s)
            return mv_normal_logpdf(
                xt, self.transition_mean(t - s, xs), cov_eigvals, self.params["eigvecs"]
            )  # take eigenvectors and eigenvalues as input in this case
        else:
            return normal_logpdf(
                xt, self.transition_mean(t - s, xs), self.transition_var(t - s)
            )

    def grad_logh(self, terminal_state, t, x):
        if self.type == "bm":
            return (
                self.invSigma * (terminal_state - x) / (self.T - t)
                - self.invSigma * self.params["alpha"]
            )

        if self.type == "ou":
            return (
                (terminal_state - self.transition_mean(self.T - t, x))
                * torch.exp(-self.params["beta"] * (self.T - t))
                / self.transition_var(self.T - t)
            )

        if self.type == "ou-full":
            cov_eigvals = self.transition_var(self.T - t)
            curr_eigvals = (1.0 / cov_eigvals) * torch.exp(
                -(self.T - t) * self.params["eigvals"]
            )
            curr_mat = (
                self.params["eigvecs"]
                @ torch.diag(curr_eigvals)
                @ self.params["eigvecs"].T
            )
            return (terminal_state - self.transition_mean(self.T - t, x)) @ curr_mat

    def log_radon_nikodym(self, trajectories, modify=True):
        N = trajectories.shape[0]
        M = trajectories.shape[1] - 1
        initial_state = trajectories[:, 0, :]
        terminal_state = trajectories[:, -1, :]
        G = torch.zeros(N, M)
        for m in range(M):
            if modify:  # time-change in Van der Meulen and Schauer (2017)
                t_current = self.time[m] * (2.0 - self.time[m] / self.T)
                stepsize = self.stepsizes[m] * 2.0 * (1.0 - self.time[m] / self.T)
            else:
                t_current = self.time[m]
                stepsize = self.stepsizes[m]
            X_current = trajectories[:, m, :]
            diff_f = self.f(t_current, X_current) - self.auxiliary_f(
                t_current, X_current
            )  # size (N, d)
            grad_logh = self.grad_logh(
                terminal_state, t_current, X_current
            )  # size (N, d)
            G[:, m] = torch.sum(diff_f * grad_logh, 1) * stepsize  # size (N)

        log_rn = torch.sum(G, 1) + self.log_transition(
            0.0, initial_state, self.T, terminal_state
        )  # size (N)

        return log_rn
