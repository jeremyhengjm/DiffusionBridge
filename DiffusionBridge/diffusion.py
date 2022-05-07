"""
A module to simulate approximations of diffusion and diffusion bridge processes.
"""

import torch    
import torch.nn.functional as F
from DiffusionBridge.neuralnet import ScoreNetwork
from DiffusionBridge.ema import ema_register, ema_update, ema_copy
from DiffusionBridge.utils import normal_logpdf

def construct_time_discretization(terminal_time, num_steps):    
    stepsizes = (terminal_time / num_steps) * torch.ones(num_steps)
    time = torch.linspace(0.0, terminal_time, num_steps + 1)
    return (time, stepsizes)

class model(torch.nn.Module):
    
    def __init__(self, f, sigma, dimension, terminal_time, num_steps):
        """
        Parameters
        ----------    
        f : drift function
        sigma : diffusion coefficient (assume constant for now)
        dimension : dimension of diffusion
        terminal_time : length of time horizon
        num_steps : number of time-discretization steps        
        """
        super().__init__()
        self.f = f
        self.sigma = sigma
        self.Sigma = sigma * sigma
        self.invSigma = 1.0 / self.Sigma
        self.d = dimension
        self.T = terminal_time
        self.num_steps = num_steps
        (self.time, self.stepsizes) = construct_time_discretization(terminal_time, num_steps)

    def simulate_process(self, initial_state, num_samples):
        """
        Simulate diffusion process using Euler-Maruyama discretization.

        Parameters
        ----------    
        initial_state : initial condition of size d

        num_samples : number of samples desired
                        
        Returns
        -------   
        output : dict containing 
            trajectories : realizations of time-discretized process (N, M+1, d)
            scaled_brownian : scaled brownian increments (N, M, d)
        """
        # initialize and preallocate
        N = num_samples
        M = self.num_steps
        X = initial_state.repeat(N, 1) # size (N, d)
        trajectories = torch.zeros(N, M+1, self.d)
        trajectories[:, 0, :] = X
        scaled_brownian = torch.zeros(N, M, self.d)
        
        # simulate process forwards in time
        for m in range(M):
            stepsize = self.stepsizes[m]
            t = self.time[m]
            drift = self.f(t, X)
            euler = X + stepsize * drift
            brownian = torch.sqrt(stepsize) * torch.randn(X.shape) # size (N x d)
            X = euler + self.sigma * brownian
            trajectories[:, m+1, :] = X
            scaled_brownian[:, m, :] = - (self.invSigma / stepsize) * self.sigma * brownian 

        # output
        output = {'trajectories' : trajectories, 'scaled_brownian' : scaled_brownian}

        return output

    def simulate_bridge_backwards(self, score_net, initial_state, terminal_state, epsilon, num_samples, modify = False):
        """
        Simulate diffusion bridge process backwards using Euler-Maruyama discretization.

        Parameters
        ----------
        score_net : neural network approximation of score function of transition density

        initial_state : initial condition of size d

        terminal_state : terminal condition of size d

        epsilon : positive constant to enforce initial constraint 

        num_samples : number of samples desired

        modify : bool specifying if variance of transitions are modified 
                        
        Returns
        -------    
        output : dict containing
            trajectories : realizations of time-discretized process (N, M+1, d)
            logdensity : log-density of simulated process (N)
            scaled_brownian : scaled brownian increments (N, M, d)
        """

        # initialize and preallocate
        N = num_samples
        M = self.num_steps
        X0 = initial_state.repeat(N, 1) # size (N, d)
        Z = terminal_state.repeat(N, 1) # size (N, d)
        trajectories = torch.zeros(N, M+1, self.d)
        trajectories[:, M, :] = Z
        logdensity = torch.zeros(N)
        scaled_brownian = torch.zeros(N, M, self.d)

        # simulate process backwards in time
        for m in range(M, 0, -1):
            stepsize = self.stepsizes[m-1]
            t = self.time[m]
            t_next = self.time[m-1]
            score = score_net(t.repeat((N,1)), Z) # size (N, d)
            drift = -self.f(t,Z) + self.Sigma * score + epsilon * (X0 - Z) / t
            euler = Z + stepsize * drift
            if (m > 1):
                if modify:
                    scaling = stepsize * t_next / t
                else:
                    scaling = stepsize
                brownian = torch.sqrt(scaling) * torch.randn(Z.shape) # size (N x d)
                Z = euler + self.sigma * brownian
                logdensity += normal_logpdf(Z, euler, scaling * self.Sigma) 
                trajectories[:, m-1, :] = Z
                scaled_brownian[:, m-1, :] = - (self.invSigma / scaling) * self.sigma * brownian
            else:
                # terminal constraint
                trajectories[:, 0, :] = X0
                scaled_brownian[:, 0, :] = - (self.invSigma / stepsize) * (X0 - euler)

        # output
        output = {'trajectories' : trajectories, 'logdensity' : logdensity, 'scaled_brownian' : scaled_brownian}
        
        return output
        

    def simulate_bridge_forwards(self, score_transition_net, score_marginal_net, initial_state, terminal_state, epsilon, num_samples, modify = False):
        """
        Simulate diffusion bridge process fowards using Euler-Maruyama discretization.

        Parameters
        ----------
        score_transition_net : neural network approximation of score function of transition density

        score_marginal_net : neural network approximation of score function of marginal (diffusion bridge) density

        initial_state : initial condition of size d

        terminal_state : terminal condition of size d

        epsilon : positive constant to enforce initial constraint 

        num_samples : number of samples desired

        modify : bool specifying if variance of transitions are modified 

        Returns
        -------   
        output : dict containing  
            trajectories : realizations of time-discretized process (N, M+1, d)
            logdensity : log-density of simulated process (N)
        """

        # initialize and preallocate
        N = num_samples
        M = self.num_steps
        T = self.T
        X = initial_state.repeat(N, 1) # size (N, d)
        X0 = initial_state.repeat(N, 1) # size (N, d)
        XT = terminal_state.repeat(N, 1) # size (N, d)
        trajectories = torch.zeros(N, M+1, self.d)
        trajectories[:, 0, :] = X
        logdensity = torch.zeros(N)
        
        # simulate process forwards in time
        for m in range(M-1):
            stepsize = self.stepsizes[m]
            if m == 0:
                t = self.time[m] + 0.5 * stepsize # fudging a little here because of singularity
            else: 
                t = self.time[m]
            t_next = self.time[m+1]
            score_marginal = score_marginal_net(t.repeat(N,1), X) # size (N, d)
            score_transition = score_transition_net(t.repeat(N,1), X) # size (N, d)
            drift = self.f(t, X) + self.Sigma * (score_marginal - score_transition) + epsilon * ((XT - X) / (self.T - t) - (X0 - X) / t)            
            euler = X + stepsize * drift
            if modify:
                scaling = stepsize * (T - t_next) / (T - t)
            else:
                scaling = stepsize            
            brownian = torch.sqrt(scaling) * torch.randn(X.shape) # size (N x d)
            X = euler + self.sigma * brownian
            logdensity += normal_logpdf(X, euler, scaling * self.Sigma)
            trajectories[:, m+1, :] = X

        # terminal constraint
        trajectories[:, M, :] = XT

        # output
        output = {'trajectories': trajectories, 'logdensity': logdensity}

        return output

    def simulate_proposal_bridge(self, drift, initial_state, terminal_state, num_samples, modify = False):
        """
        Simulate proposal diffusion bridge process fowards using Euler-Maruyama discretization.

        Parameters
        ----------
        drift : proposal drift function

        initial_state : initial condition of size d

        terminal_state : terminal condition of size d

        num_samples : number of samples desired

        modify : bool specifying if variance of transitions are modified 

        Returns
        -------   
        output : dict containing  
            trajectories : realizations of time-discretized process (N, M+1, d)
            logdensity : log-density of simulated process (N)
        """

        # initialize and preallocate
        N = num_samples
        M = self.num_steps
        T = self.T
        X = initial_state.repeat(N, 1) # size (N, d)
        XT = terminal_state.repeat(N, 1) # size (N, d)
        trajectories = torch.zeros(N, M+1, self.d)
        trajectories[:, 0, :] = X
        logdensity = torch.zeros(N)
        
        # simulate process forwards in time
        for m in range(M-1):
            stepsize = self.stepsizes[m]
            t = self.time[m]
            t_next = self.time[m+1]
            euler = X + stepsize * drift(t, X)
            if modify:
                scaling = stepsize * (T - t_next) / (T - t)
            else:
                scaling = stepsize
            brownian = torch.sqrt(scaling) * torch.randn(X.shape) # size (N x d)
            X = euler + self.sigma * brownian
            logdensity += normal_logpdf(X, euler, scaling * self.Sigma)
            trajectories[:, m+1, :] = X

        # terminal constraint
        trajectories[:, M, :] = XT

        # output
        output = {'trajectories': trajectories, 'logdensity': logdensity}

        return output

    def law_bridge(self, trajectories):
        """
        Evaluate law of (time-discretized) diffusion bridge process.

        Parameters
        ----------
        trajectories : realizations of time-discretized proposal bridge process satisfying initial and termianl constraints (N, M+1, d)

        Returns
        -------    
        logdensity : log-density values of size N
        """

        N = trajectories.shape[0]
        M = self.num_steps
        logdensity = torch.zeros(N)

        for m in range(M):
            stepsize = self.stepsizes[m]
            t = self.time[m]
            X_current = trajectories[:, m, :]
            drift = self.f(t, X_current)
            euler = X_current + stepsize * drift
            X_next = trajectories[:, m+1, :]
            logdensity += normal_logpdf(X_next, euler, stepsize * self.Sigma)

        return logdensity

    def gradient_transition(self, trajectories, scaled_brownian, epsilon):
        """
        Evaluate gradient function needed in score matching to learn score of transition density.

        Parameters
        ----------
        trajectories : realizations of time-discretized process (N, M+1, d)

        scaled_brownian : scaled brownian increments (N, M, d)

        epsilon : positive constant to enforce initial constraint 

        Returns
        -------    
        grad : gradient function evaluations (N, M, d)
        """
        N = trajectories.shape[0]
        M = self.num_steps
        grad = torch.zeros(N, M, self.d)
        X0 = trajectories[:, 0, :] 

        for m in range(M):
            X_next = trajectories[:, m+1, :]
            t_current = self.time[m]
            t_next = self.time[m+1]
            grad[:, m, :] = scaled_brownian[:, m, :] - epsilon * self.invSigma * (X0 - X_next) / (self.T - t_next)            
            if (m == (M-1)):
                # fudging a little here because of singularity
                grad[:, m, :] = scaled_brownian[:, m, :] - epsilon * self.invSigma * (X0 - X_next) / (self.T - t_current) 

        return grad

    def learn_score_transition(self, initial_state, terminal_state, epsilon, minibatch, num_iterations, learning_rate, ema_momentum):
        """
        Learn approximation of score transition using score matching.

        Parameters
        ----------
        initial_state : initial condition of size d

        terminal_state : terminal condition of size d

        epsilon : positive constant to enforce initial constraint 

        minibatch : number of mini-batch samples desired

        num_iterations : number of optimization iterations (divisible by num_batches)

        learning_rate : learning rate of Adam optimizer

        ema_momentum : momentum parameter of exponential moving average update
                        
        Returns
        -------
        output : dict containing    
            net : neural network approximation of score function of transition density
            loss : value of loss function during learning
        """

        M = self.num_steps
        N = minibatch
        timesteps = self.time[1:(M+1)].reshape((1,M,1)).repeat((N,1,1)) # size (N, M, 1)
        timesteps_flatten = timesteps.flatten(start_dim = 0, end_dim = 1) # size (N*M, 1)
        loss_values = torch.zeros(num_iterations)

        # create score network
        score_net = ScoreNetwork(dimension = self.d)
        ema_parameters = ema_register(score_net)

        # optimization
        optimizer = torch.optim.Adam(score_net.parameters(), lr = learning_rate)
        num_batches = 10 
        num_samples = num_batches * N
        num_repeats = int(num_iterations / num_batches)
        iteration = 1
        for i in range(num_repeats):            
            # simulate trajectories from diffusion process
            simulation_output = self.simulate_process(initial_state, num_samples)
            trajectories = simulation_output['trajectories']
            scaled_brownian = simulation_output['scaled_brownian']

            for j in range(num_batches):
                # get minibatch of trajectories
                traj = trajectories[(j * N):((j+1) * N), :, :] # size (N, M+1, d)
                scaled = scaled_brownian[(j * N):((j+1) * N), :, :]

                # evaluate gradient function
                grad = self.gradient_transition(traj, scaled, epsilon) # size (N, M, d)
                grad_flatten = grad.flatten(start_dim = 0, end_dim = 1) # size (N*M, d)
                
                # evaluate score network
                traj_flatten = traj[:, 1:(M+1), :].flatten(start_dim = 0, end_dim = 1) # size (N*M, d)
                score = score_net(timesteps_flatten, traj_flatten) # size (N*M, d)

                # compute loss function
                loss = F.mse_loss(score, grad_flatten) # need to extend this for non-uniform stepsizes
            
                # backpropagation
                loss.backward()
 
                # optimization step and zero gradient
                optimizer.step()
                optimizer.zero_grad()

                # update parameters using exponential moving average
                ema_update(ema_parameters, score_net, ema_momentum)

                # iteration counter
                current_loss = loss.item()
                loss_values[iteration-1] = current_loss
                if (iteration == 1) or (iteration % 50 == 0):
                    print("Optimization iteration:", iteration, "Loss:", current_loss)
                iteration += 1

        # use exponential moving average parameters in score network
        ema_copy(ema_parameters, score_net)           

        # output 
        output = {'net' : score_net, 'loss' : loss_values}
        
        return output
    
    def gradient_marginal(self, trajectories, scaled_brownian, epsilon):
        """
        Evaluate gradient function needed in score matching to learn score of marginal (diffusion bridge) density.

        Parameters
        ----------
        trajectories : realizations of time-discretized process (N, M+1, d)

        scaled_brownian : scaled brownian increments (N, M, d)

        epsilon : positive constant to enforce initial constraint 

        Returns
        -------    
        grad : gradient function evaluations (N, M, d)
        """
        N = trajectories.shape[0]
        M = self.num_steps
        grad = torch.zeros(N, M, self.d)
        XT = trajectories[:, M, :] 

        for m in range(M,0,-1):
            Z_next = trajectories[:, m-1, :,]
            t_current = self.time[m]
            t_next = self.time[m-1]
            grad[:, m-1, :] = scaled_brownian[:, m-1, :] - epsilon * self.invSigma * (XT - Z_next) / (self.T - t_next)            
            if (m == 1):
                # fudging a little here because of singularity
                grad[:, 0, :] = scaled_brownian[:, 0, :] - epsilon * self.invSigma * (XT - Z_next) / (self.T - t_current) 

        return grad

    def learn_score_marginal(self, score_transition_net, initial_state, terminal_state, epsilon, minibatch, num_iterations, learning_rate, ema_momentum):
        """
        Learn approximation of score of marginal (diffusion bridge) density using score matching.

        Parameters
        ----------
        score_transition_net : neural network approximation of score function of transition density

        initial_state : initial condition of size d

        terminal_state : terminal condition of size d

        epsilon : positive constant to enforce initial constraint 

        minibatch : number of mini-batch samples desired

        num_iterations : number of optimization iterations (divisible by num_batches)

        learning_rate : learning rate of Adam optimizer

        ema_momentum : momentum parameter of exponential moving average update
                        
        Returns
        -------
        output : dict containing     
            score_marginal_net : neural network approximation of score function of marginal (diffusion bridge) density
            loss_values : value of loss function during learning
        """

        M = self.num_steps
        N = minibatch
        timesteps = self.time[0:M].reshape((1,M,1)).repeat((N,1,1)) # size (N, M, 1)
        timesteps_flatten = timesteps.flatten(start_dim = 0, end_dim = 1) # size (N*M, 1)
        loss_values = torch.zeros(num_iterations)

        # create score network
        score_net = ScoreNetwork(dimension = self.d)
        ema_parameters = ema_register(score_net)

        # optimization
        optimizer = torch.optim.Adam(score_net.parameters(), lr = learning_rate)
        num_batches = 10 
        num_samples = num_batches * N
        num_repeats = int(num_iterations / num_batches)
        iteration = 1
        for i in range(num_repeats):            
            # simulate trajectories from approximate diffusion bridge process backwards
            with torch.no_grad():
                simulation_output = self.simulate_bridge_backwards(score_transition_net, initial_state, terminal_state, epsilon, num_samples, modify = True)            
            trajectories = simulation_output['trajectories']
            scaled_brownian = simulation_output['scaled_brownian']

            for j in range(num_batches):
                # get minibatch of trajectories
                traj = trajectories[(j * N):((j+1) * N), :, :] # size (N, M+1, d)
                scaled = scaled_brownian[(j * N):((j+1) * N), :, :]

                # evaluate gradient function
                grad = self.gradient_marginal(traj, scaled, epsilon) # size (N, M, d)
                grad_flatten = grad.flatten(start_dim = 0, end_dim = 1) # size (N*M, d)
                
                # evaluate score network
                traj_flatten = traj[:, 0:M, :].flatten(start_dim = 0, end_dim = 1) # size (N*M, d)
                score = score_net(timesteps_flatten, traj_flatten) # size (N*M, d)

                # compute loss function
                loss = F.mse_loss(score, grad_flatten) # need to extend this for non-uniform stepsizes
            
                # backpropagation
                loss.backward()
 
                # optimization step and zero gradient
                optimizer.step()
                optimizer.zero_grad()

                # update parameters using exponential moving average
                ema_update(ema_parameters, score_net, ema_momentum)

                # iteration counter
                current_loss = loss.item()
                loss_values[iteration-1] = current_loss
                if (iteration == 1) or (iteration % 50 == 0):
                    print("Optimization iteration:", iteration, "Loss:", current_loss)
                iteration += 1

        # use exponential moving average parameters in score network
        ema_copy(ema_parameters, score_net)           

        # output 
        output = {'net' : score_net, 'loss' : loss_values}

        return output
    