"""
A module to simulate approximations of diffusion and diffusion bridge processes.
"""

import torch    
import torch.nn.functional as F
from DiffusionBridge.neuralnet import ScoreNetwork, FullScoreNetwork
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

    def simulate_process(self, initial_states):
        """
        Simulate diffusion process using Euler-Maruyama discretization.

        Parameters
        ----------    
        initial_states : initial condition of size (N, d)
                        
        Returns
        -------   
        output : dict containing 
            trajectories : realizations of time-discretized process (N, M+1, d)
            scaled_brownian : scaled brownian increments (N, M, d)
        """
        # initialize and preallocate
        N = initial_states.shape[0]
        M = self.num_steps
        X = initial_states.clone() # size (N ,d)
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

    def simulate_bridge_backwards(self, score_net, initial_state, terminal_state, epsilon, num_samples = 1, modify = False, full_score = False, new_num_steps = None):
        """
        Simulate diffusion bridge process backwards using Euler-Maruyama discretization.

        Parameters
        ----------
        score_net : neural network approximation of score function of transition density

        initial_state : initial condition of size d or (N, d)

        terminal_state : terminal condition of size d or (N, d)

        epsilon : positive constant to enforce initial constraint 

        num_samples : number of samples desired

        modify : bool specifying if variance of transitions are modified 

        full_score : bool specifying if the full score function is employed

        new_num_steps : new number of time-discretization steps        
                        
        Returns
        -------    
        output : dict containing
            trajectories : realizations of time-discretized process (N, M+1, d)
            logdensity : log-density of simulated process (N)
            scaled_brownian : scaled brownian increments (N, M, d)
            score_evaluations : evaluations of score network (N, M, d)
        """

        # initialize and preallocate
        if len(initial_state.shape) == 1:
            X0 = initial_state.repeat(num_samples, 1) # size (N, d)
            N = num_samples
        else:
            X0 = initial_state.clone() 
            N = initial_state.shape[0]        
        
        if len(terminal_state.shape) == 1:
            Z = terminal_state.repeat(num_samples, 1) # size (N, d)
            N = num_samples
        else:
            Z = terminal_state.clone()
            N = terminal_state.shape[0]        
        
        if new_num_steps is None:
            M = self.num_steps
            timesteps = self.time
            stepsizes = self.stepsizes
        else: 
            M = new_num_steps
            (timesteps, stepsizes) = construct_time_discretization(self.T, M)

        trajectories = torch.zeros(N, M+1, self.d)
        trajectories[:, M, :] = Z
        logdensity = torch.zeros(N)
        scaled_brownian = torch.zeros(N, M, self.d)
        score_evaluations = torch.zeros(N, M, self.d)

        # simulate process backwards in time
        for m in range(M, 0, -1):
            stepsize = stepsizes[m-1]
            t = timesteps[m]
            t_next = timesteps[m-1]
            if full_score:
                score = score_net(t.repeat((N,1)), Z, X0) # size (N, d)
            else:
                score = score_net(t.repeat((N,1)), Z) # size (N, d)
            score_evaluations[:, m-1, :] = score
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
                if modify:
                    # fudging a little here because of singularity
                    scaling = stepsize * 0.25 * stepsize / t
                else:
                    scaling = stepsize
                trajectories[:, 0, :] = X0
                scaled_brownian[:, 0, :] = - (self.invSigma / scaling) * (X0 - euler)

        # output
        output = {'trajectories' : trajectories, 'logdensity' : logdensity, 'scaled_brownian' : scaled_brownian, 'score_evaluations' : score_evaluations}
        
        return output
        

    def simulate_bridge_forwards(self, score_transition_net, score_marginal_net, initial_state, terminal_state, epsilon, num_samples = 1, modify = False, full_score = False, new_num_steps = None):
        """
        Simulate diffusion bridge process fowards using Euler-Maruyama discretization.

        Parameters
        ----------
        score_transition_net : neural network approximation of score function of transition density

        score_marginal_net : neural network approximation of score function of marginal (diffusion bridge) density

        initial_state : initial condition of size d or (N, d)

        terminal_state : terminal condition of size d or (N, d)

        epsilon : positive constant to enforce initial constraint 

        num_samples : number of samples desired

        modify : bool specifying if variance of transitions are modified 

        full_score : bool specifying if the full score function is employed

        new_num_steps : new number of time-discretization steps        

        Returns
        -------   
        output : dict containing  
            trajectories : realizations of time-discretized process (N, M+1, d)
            logdensity : log-density of simulated process (N)
        """

        # initialize and preallocate
        T = self.T
        if len(initial_state.shape) == 1:
            X0 = initial_state.repeat(num_samples, 1) # size (N, d)
            X = initial_state.repeat(num_samples, 1) # size (N, d)
            N = num_samples
        else:
            X0 = initial_state.clone()
            X = initial_state.clone() 
            N = initial_state.shape[0]        
        
        if len(terminal_state.shape) == 1:
            XT = terminal_state.repeat(num_samples, 1) # size (N, d)
            N = num_samples
        else:
            XT = terminal_state.clone()
            N = terminal_state.shape[0]
        
        if new_num_steps is None:
            M = self.num_steps
            timesteps = self.time
            stepsizes = self.stepsizes
        else: 
            M = new_num_steps
            (timesteps, stepsizes) = construct_time_discretization(T, M)

        trajectories = torch.zeros(N, M+1, self.d)
        trajectories[:, 0, :] = X
        logdensity = torch.zeros(N)
        
        # simulate process forwards in time
        for m in range(M-1):
            stepsize = stepsizes[m]
            if m == 0:
                # fudging a little here because of singularity                
                t = timesteps[m] + 0.5 * stepsize 
            else: 
                t = timesteps[m]
            t_next = timesteps[m+1]
            if full_score:
                score_marginal = score_marginal_net(t.repeat(N,1), X, X0, XT) # size (N, d)
                score_transition = score_transition_net(t.repeat(N,1), X, X0) # size (N, d)
            else:    
                score_marginal = score_marginal_net(t.repeat(N,1), X) # size (N, d)
                score_transition = score_transition_net(t.repeat(N,1), X) # size (N, d)
            drift = self.f(t, X) + self.Sigma * (score_marginal - score_transition) + epsilon * ((XT - X) / (T - t) - (X0 - X) / t)            
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

    def simulate_proposal_bridge(self, drift, initial_state, terminal_state, num_samples, modify = False, new_num_steps = None):
        """
        Simulate proposal diffusion bridge process fowards using Euler-Maruyama discretization.

        Parameters
        ----------
        drift : proposal drift function

        initial_state : initial condition of size d

        terminal_state : terminal condition of size d

        num_samples : number of samples desired

        modify : bool specifying if variance of transitions are modified 

        new_num_steps : new number of time-discretization steps        

        Returns
        -------   
        output : dict containing  
            trajectories : realizations of time-discretized process (N, M+1, d)
            logdensity : log-density of simulated process (N)
        """

        # initialize and preallocate
        N = num_samples
        T = self.T

        if new_num_steps is None:
            M = self.num_steps
            timesteps = self.time
            stepsizes = self.stepsizes
        else: 
            M = new_num_steps
            (timesteps, stepsizes) = construct_time_discretization(T, M)
        
        X = initial_state.repeat(N, 1) # size (N, d)
        XT = terminal_state.repeat(N, 1) # size (N, d)
        trajectories = torch.zeros(N, M+1, self.d)
        trajectories[:, 0, :] = X
        logdensity = torch.zeros(N)
        
        # simulate process forwards in time
        for m in range(M-1):
            stepsize = stepsizes[m]
            t = timesteps[m]
            t_next = timesteps[m+1]
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

    def law_bridge(self, trajectories, new_num_steps = None):
        """
        Evaluate law of (time-discretized) diffusion bridge process.

        Parameters
        ----------
        trajectories : realizations of time-discretized proposal bridge process satisfying initial and terminal constraints (N, M+1, d)

        new_num_steps : new number of time-discretization steps

        Returns
        -------    
        logdensity : log-density values of size N
        """

        N = trajectories.shape[0]
        logdensity = torch.zeros(N)

        if new_num_steps is None:
            M = self.num_steps
            timesteps = self.time
            stepsizes = self.stepsizes
        else: 
            M = new_num_steps
            (timesteps, stepsizes) = construct_time_discretization(self.T, M)
        
        for m in range(M):
            stepsize = stepsizes[m]
            t = timesteps[m]
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
            if (m == (M-1)):
                # fudging a little here because of singularity                
                t_next = self.time[m+1] - 0.25 * self.stepsizes[m]
            else: 
                t_next = self.time[m+1]
            grad[:, m, :] = scaled_brownian[:, m, :] - epsilon * self.invSigma * (X0 - X_next) / (self.T - t_next)            

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
            initial_states = initial_state.repeat(num_samples, 1) # size (N, d)
            simulation_output = self.simulate_process(initial_states)
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

    def learn_full_score_transition(self, simulate_initial_state, terminal_state, epsilon, minibatch, num_initial_per_batch, num_iterations, learning_rate, ema_momentum):
        """
        Learn full approximation of score transition using score matching.

        Parameters
        ----------
        simulate_initial_state : function returning simulated initial conditions

        terminal_state : terminal condition of size d

        epsilon : positive constant to enforce initial constraint 

        minibatch : number of mini-batch samples desired

        num_initial_per_batch : number of initial states per batch

        num_iterations : number of optimization iterations (divisible by num_batches)

        learning_rate : learning rate of Adam optimizer

        ema_momentum : momentum parameter of exponential moving average update
                        
        Returns
        -------
        output : dict containing    
            net : neural network approximation of full score function of transition density
            loss : value of loss function during learning
        """

        d = self.d
        M = self.num_steps
        N = minibatch * num_initial_per_batch
        timesteps = self.time[1:(M+1)].reshape((1,M,1)).repeat((N,1,1)) # size (N, M, 1)
        timesteps_flatten = timesteps.flatten(start_dim = 0, end_dim = 1) # size (N*M, 1)        
        loss_values = torch.zeros(num_iterations)

        # create score network
        score_net = FullScoreNetwork(dimension = self.d)
        ema_parameters = ema_register(score_net)

        # optimization
        optimizer = torch.optim.Adam(score_net.parameters(), lr = learning_rate)
        for i in range(num_iterations):
            # simulate initial states
            initial_states = simulate_initial_state(num_initial_per_batch).repeat((minibatch,1)) # size (N, d)
            initial_states_repeated = initial_states.reshape((N,1,d)).repeat((1,M,1)) # size (N, M, d)
            initial_states_flatten = initial_states_repeated.flatten(start_dim = 0, end_dim = 1) # size (N*M, d)

            # simulate trajectories from diffusion process
            simulation_output = self.simulate_process(initial_states)
            traj = simulation_output['trajectories'] # size (N, M+1, d)
            scaled = simulation_output['scaled_brownian']

            # evaluate gradient function
            grad = self.gradient_transition(traj, scaled, epsilon) # size (N, M, d)
            grad_flatten = grad.flatten(start_dim = 0, end_dim = 1) # size (N*M, d)
                
            # evaluate score network
            traj_flatten = traj[:, 1:(M+1), :].flatten(start_dim = 0, end_dim = 1) # size (N*M, d)
            score = score_net(timesteps_flatten, traj_flatten, initial_states_flatten) # size (N*M, d)

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
            loss_values[i] = current_loss
            if (i == 0) or ((i+1) % 50 == 0):
                print("Optimization iteration:", i+1, "Loss:", current_loss)

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
            if (m == 1):
                # fudging a little here because of singularity                
                t_next = 0.25 * self.stepsizes[m-1]
            else: 
                t_next = self.time[m-1]
            grad[:, m-1, :] = scaled_brownian[:, m-1, :] - epsilon * self.invSigma * (XT - Z_next) / t_next

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
                simulation_output = self.simulate_bridge_backwards(score_transition_net, initial_state, terminal_state, epsilon, num_samples, modify = True, full_score = False)            
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
        