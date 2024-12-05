import torch
import torch.nn as nn
import random
import numpy as np

class BayesFittingNet(nn.Module):
    def __init__(self, obs, mu_likelihood, mu_posterior, device, diag_only=False):
        super().__init__()
        self.device = device
        
        # Constants from original code
        self.cov_max = 10
        self.cov_min = 1e-2
        
        # Convert inputs to tensors and move to GPU once
        self.obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
        self.mu_likelihood = torch.tensor(mu_likelihood, dtype=torch.float32, device=self.device)
        self.mu_posterior = torch.tensor(mu_posterior, dtype=torch.float32, device=self.device)
        self.diag_only = diag_only
        
        # Initialize parameters matching the original initialization
        mu_prior_init = torch.zeros(3, device=self.device)
        mu_prior_init[0] = torch.tensor(np.random.uniform(15.0, 25.0))
        mu_prior_init[1] = torch.tensor(np.random.uniform(15.0, 25.0))
        mu_prior_init[2] = torch.tensor(np.random.uniform(-np.pi*(1/16), np.pi*(1/16)))
        self.mu_prior_pose = nn.Parameter(mu_prior_init)
        
        # Initialize covariance matrices
        self.dim = 16 if self.diag_only else 32
        if self.diag_only:
            Sigma_prior_init = torch.tensor(np.random.uniform(self.cov_min, self.cov_max, 16), device=self.device)
            Sigma_likelihood_init = torch.tensor(np.random.uniform(self.cov_min, self.cov_max, 16), device=self.device)
            
            self.Sigma_prior_params = nn.Parameter(Sigma_prior_init)
            self.Sigma_likelihood_params = nn.Parameter(Sigma_likelihood_init)
        else:
            Sigma_prior_init = torch.tensor(np.random.uniform(self.cov_min, self.cov_max, 32), device=self.device)
            Sigma_likelihood_init = torch.tensor(np.random.uniform(self.cov_min, self.cov_max, 32), device=self.device)
            
            # Apply the same covariance constraints as original
            for i in range(8):
                b = i * 4
                Sigma_prior_init[b+1] = torch.minimum(Sigma_prior_init[b], Sigma_prior_init[b+3])/2
                Sigma_prior_init[b+2] = torch.minimum(Sigma_prior_init[b], Sigma_prior_init[b+3])/2
                Sigma_likelihood_init[b+1] = torch.minimum(Sigma_likelihood_init[b], Sigma_likelihood_init[b+3])/2
                Sigma_likelihood_init[b+2] = torch.minimum(Sigma_likelihood_init[b], Sigma_likelihood_init[b+3])/2
            
            self.Sigma_prior_params = nn.Parameter(Sigma_prior_init)
            self.Sigma_likelihood_params = nn.Parameter(Sigma_likelihood_init)
        
        self.iteration_count = 0

    def construct_covariance_matrices(self):
        if self.diag_only:
            # Remove exponential transform to match original
            Sigma_prior = torch.diag(self.Sigma_prior_params)
            Sigma_likelihood = torch.diag(self.Sigma_likelihood_params)
        else:
            # Construct block diagonal matrices
            Sigma_prior = self._build_block_diagonal(self.Sigma_prior_params)
            Sigma_likelihood = self._build_block_diagonal(self.Sigma_likelihood_params)
        
        return Sigma_prior, Sigma_likelihood

    def _build_block_diagonal(self, params):
        matrix = torch.zeros((16, 16), device=self.device)
        for i in range(8):
            j = i * 2
            k = i * 4
            block = params[k:k+4].reshape(2, 2)
            # Ensure positive definiteness
            block = block @ block.T + torch.eye(2, device=self.device) * 1e-6
            matrix[j:j+2, j:j+2] = block
        return matrix
    
    def transform_mu_gpu(self, mu, pose, version=2):
        # pose is in the form [x, y, theta]
        # mu is in the form [x1, x2, ..., x8, y1, y2, ..., y8] version 1 or [x1, y1, x2, y2, ..., x8, y8] version 2
        
        # Reshape points into 2xN array for easier manipulation
        if version == 1:
            points = torch.vstack([mu[:8], mu[8:]])  # Shape: (2, 8)
        elif version == 2:
            x_points = mu[0::2]  # Every even index
            y_points = mu[1::2]  # Every odd index
            points = torch.vstack([x_points, y_points])  # Shape: (2, 8)
        
        # Calculate centroid
        centroid = torch.mean(points, dim=1, keepdim=True)  # Shape: (2, 1)
        
        # Create rotation matrix with requires_grad maintained
        rotation_matrix = torch.tensor([
            [torch.cos(pose[2]), -torch.sin(pose[2])],
            [torch.sin(pose[2]), torch.cos(pose[2])]
        ], device=self.device, dtype=torch.float32)
        
        # Center points, rotate, then translate back and add pose translation
        rel_points = points - centroid
        rel_points = rotation_matrix @ rel_points
        points = rel_points + pose[:2].unsqueeze(1)
        
        # Return in original format 
        if version == 1:
            # [x1, ..., x8, y1, ..., y8]
            return torch.cat((points[0], points[1]))
        elif version == 2:
            # [x1, y1, x2, y2, ..., x8, y8]
            mu_new = torch.zeros(16, device=mu.device)
            mu_new[0::2] = points[0]  # Even indices get x coordinates
            mu_new[1::2] = points[1]  # Odd indices get y coordinates
            return mu_new

    def forward(self):
        self.iteration_count += 1
        
        Sigma_prior, Sigma_likelihood = self.construct_covariance_matrices()
        
        try:
            # Compute posterior using Cholesky decomposition for better stability
            eye = torch.eye(16, device=self.device, dtype=torch.float32)
            
            # Compute prior precision using Cholesky
            L_prior = torch.linalg.cholesky(Sigma_prior)
            prior_precision = torch.cholesky_solve(eye, L_prior)
            
            # Compute likelihood precision using Cholesky
            L_likelihood = torch.linalg.cholesky(Sigma_likelihood)
            likelihood_precision = torch.cholesky_solve(eye, L_likelihood)
            
            # Compute posterior precision and its Cholesky factor
            posterior_precision = prior_precision + likelihood_precision
            L_posterior = torch.linalg.cholesky(posterior_precision)
            
            # Compute posterior covariance using Cholesky solve
            Sigma_posterior_pred = torch.cholesky_solve(eye, L_posterior)
            
            # Transform mu_prior using the pose parameters
            mu_prior = self.transform_mu_gpu(self.mu_likelihood, self.mu_prior_pose, version=2)
            
            # Compute predicted posterior mean using Cholesky solves
            mu_posterior_pred = Sigma_posterior_pred @ (
                torch.cholesky_solve(mu_prior.unsqueeze(1), L_prior).squeeze(1) + 
                torch.cholesky_solve(self.mu_likelihood.unsqueeze(1), L_likelihood).squeeze(1)
            )
            
            # Compute loss
            diff = self.obs - mu_posterior_pred
            logdet = 2 * torch.sum(torch.log(torch.diagonal(L_posterior)))  # More stable than torch.logdet
            quad_terms = torch.sum((diff @ torch.cholesky_solve(eye, L_posterior)) * diff, dim=1)
            
            log_terms = -0.5 * self.dim * torch.log(torch.tensor(2 * torch.pi, dtype=torch.float32, device=self.device)) - 0.5 * logdet - 0.5 * quad_terms
            loss = -torch.sum(log_terms)
            
            if self.iteration_count % 100 == 0:
                print(f"Iteration {self.iteration_count:4d} | Loss: {loss.item():10.4f}")
            
            return loss
            
        except RuntimeError as e:
            print(f"Error in forward pass: {e}")  # Add error printing
            return torch.tensor(1e10, device=self.device, requires_grad=True)  # Make sure fallback has grad

