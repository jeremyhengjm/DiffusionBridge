# Simulating Diffusion Bridges with Score Matching
This is a PyTorch implementation of the methodology described in the article [arXiv:2111.07243](https://arxiv.org/abs/2111.07243).

Run the following Python notebooks to repeat experiments for each model:
1. `repeat_ou.ipynb` for Ornstein-Uhlenbeck process;
2. `repeat_radial.ipynb` for a specific case of an interest rates model proposed by [Aït-Sahalia and Lo (Journal of Finance, 1998)](https://doi.org/10.1111/0022-1082.215228);
3. `repeat_cell.ipynb` for cell differentiation and development model of [Wang et al. (PNAS, 2011)](https://www.pnas.org/doi/abs/10.1073/pnas.1017017108).

The neural network architectures for all our numerical examples is illustrated in `neuralnet.pdf`.