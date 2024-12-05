import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy import linalg
import torch
import torch.nn as nn
# Add at the top of the file, after imports
iteration_count = 0

def check_cov_for_nan(cov):
    if np.isnan(cov).any():
        return False
    else:
        return True

def transform_mu(mu, pose, version=2):
    # pose is in the form [x, y, theta]
    # mu is in the form [x1, x2, ..., x8, y1, y2, ..., y8] version 1 or [x1, y1, x2, y2, ..., x8, y8] version 2
    
    # Reshape points into 2xN array for easier manipulation
    if version == 1:
        points = np.vstack([mu[:8], mu[8:]])  # Shape: (2, 8)
    elif version == 2:
        x_points = [mu[i] for i in range(0, 16, 2)]
        y_points = [mu[i] for i in range(1, 16, 2)]
        points = np.vstack([x_points, y_points])  # Shape: (2, 8)
    
    # Calculate centroid
    centroid = np.mean(points, axis=1, keepdims=True)  # Shape: (2, 1)
    
    # Create rotation matrix
    rotation_matrix = np.array([
        [np.cos(pose[2]), -np.sin(pose[2])],
        [np.sin(pose[2]), np.cos(pose[2])]
    ])
    
    # Center points, rotate, then translate back and add pose translation
    rel_points = points - centroid
    rel_points = rotation_matrix @ rel_points
    points = rel_points + pose[:2].reshape(-1, 1)
    
    # Return in original format 
    if version == 1:
        # [x1, ..., x8, y1, ..., y8]
        return np.concatenate((points[0], points[1]))
    elif version == 2:
        # [x1, y1, x2, y2, ..., x8, y8]
        mu_new = np.zeros(16)
        for i in range(8):
            mu_new[i*2] = points[0, i]
            mu_new[i*2+1] = points[1, i]
        return mu_new

def prior_objective(connection_key, mu_prior, mu_likelihood):
    # Assumes the distances between the points are alredy known and stay the same for the prior.
    # This function keeps the shape of the prior the same as the likelihood.

    # Prior in the form [x1, x2, ..., x8, y1, y2, ..., y8]
    loss = 0
    for connection in connection_key:
        likelihood_dist = np.linalg.norm(
            np.array([mu_likelihood[connection[0]-1], mu_likelihood[connection[0]+7]]) - 
            np.array([mu_likelihood[connection[1]-1], mu_likelihood[connection[1]+7]])
        )
        prior_dist = np.linalg.norm(
            np.array([mu_prior[connection[0]-1], mu_prior[connection[0]+7]]) - 
            np.array([mu_prior[connection[1]-1], mu_prior[connection[1]+7]])
        )
        loss += (likelihood_dist - prior_dist)**2
    return loss

def mu_loss(mu_pred, mu_true):
    loss = 0
    for i in range(len(mu_pred)//2):
        loss += np.linalg.norm(np.array([mu_pred[i], mu_pred[i+8]]) - np.array([mu_true[i], mu_true[i+8]]))**2
    return loss

def same_covariance(params):
    mu_prior_pose = params[:3]
    Sigma_prior_small = params[3:7]
    Sigma_likelihood_small = params[7:]

    Sigma_prior_small_2 = np.array([[Sigma_prior_small[0], Sigma_prior_small[1]], 
                                    [Sigma_prior_small[2], Sigma_prior_small[3]]])
    Sigma_likelihood_small_2 = np.array([[Sigma_likelihood_small[0], Sigma_likelihood_small[1]], 
                                         [Sigma_likelihood_small[2], Sigma_likelihood_small[3]]])

    Sigma_prior = np.zeros((16, 16))
    Sigma_likelihood = np.zeros((16, 16))

    # Propagate the 2x2 matrices diagonally across the 16x16 matrices
    for i in range(0, 16, 2):
        Sigma_prior[i:i+2, i:i+2] = Sigma_prior_small_2
        Sigma_likelihood[i:i+2, i:i+2] = Sigma_likelihood_small_2
    return mu_prior_pose, Sigma_prior, Sigma_likelihood

def individual_covariance(params):
    mu_prior_pose = params[:3]
    Sigma_prior_small = params[3:3 + 4*8] # 32 elements
    Sigma_likelihood_small = params[3 + 4*8:] # 32 elements

    Sigma_prior = np.zeros((16, 16))
    Sigma_likelihood = np.zeros((16, 16))

    for i in range(8):
        j = i*2
        k = i*4
        Sigma_prior[j, j] = Sigma_prior_small[k]
        Sigma_prior[j, j+1] = Sigma_prior_small[k+1]
        Sigma_prior[j+1, j] = Sigma_prior_small[k+2]
        Sigma_prior[j+1, j+1] = Sigma_prior_small[k+3]
        Sigma_likelihood[j, j] = Sigma_likelihood_small[k]
        Sigma_likelihood[j, j+1] = Sigma_likelihood_small[k+1]
        Sigma_likelihood[j+1, j] = Sigma_likelihood_small[k+2]
        Sigma_likelihood[j+1, j+1] = Sigma_likelihood_small[k+3]
    
    return mu_prior_pose, Sigma_prior, Sigma_likelihood

def identity_covariance(params):
    mu_prior_pose = params[:3]
    Sigma_prior = np.diag(params[3:16+3])
    Sigma_likelihood = np.diag(params[16+3:])
    return mu_prior_pose, Sigma_prior, Sigma_likelihood

def extract_individual_covariance(Sigma_prior_opt, Sigma_likelihood_opt, i):
    prior_block = Sigma_prior_opt[i*4:(i+1)*4]
    likelihood_block = Sigma_likelihood_opt[i*4:(i+1)*4]
    
    # Reshape 1D arrays of 4 elements into 2x2 matrices
    prior_matrix = np.array([[prior_block[0], prior_block[1]],
                            [prior_block[2], prior_block[3]]])
    likelihood_matrix = np.array([[likelihood_block[0], likelihood_block[1]],
                                [likelihood_block[2], likelihood_block[3]]])
    
    return prior_matrix, likelihood_matrix

def mu_reorganize(mu, version):
    if version == 1:
        # [x1, x2, ..., x8, y1, y2, ..., y8] -> [x1, y1, x2, y2, ..., x8, y8]
        mu_new = np.zeros(16)
        for i in range(8):
            mu_new[i*2] = mu[i]
            mu_new[i*2+1] = mu[i+8]
    elif version == 2:
        # [x1, y1, x2, y2, ..., x8, y8] -> [x1, x2, ..., x8, y1, y2, ..., y8]
        mu_new = np.zeros(16)
        for i in range(8):
            mu_new[i] = mu[i*2]
            mu_new[i+8] = mu[i*2+1]
    return mu_new
        

def general_objective(params, obs, mu_likelihood, mu_posterior, trial_number, diag_only=False):

    # Add global counter
    global iteration_count
    iteration_count += 1

    # Dimensions
    N, dim = obs.shape

    # Unpack params and create positive definite matrices using Cholesky
    # try:

    if not diag_only:
        mu_prior_pose, Sigma_prior, Sigma_likelihood = individual_covariance(params)
    else:
        # mu_prior_pose, Sigma_prior, Sigma_likelihood = same_covariance(params) # Projects the covariance to be the same for all points
        mu_prior_pose, Sigma_prior, Sigma_likelihood = identity_covariance(params) # Diagonal covariance
    # np.set_printoptions(precision=2, suppress=True, linewidth=120)
    # print(f"Sigma_prior:\n{Sigma_prior}")

    # # Compute the posterior covariance
    # Sigma_posterior_pred = linalg.inv(linalg.inv(Sigma_prior) + linalg.inv(Sigma_likelihood))
    
    try:
        # Compute posterior using cho_solve for better stability
        prior_precision = linalg.cho_solve(
            linalg.cho_factor(Sigma_prior), 
            np.eye(dim)
        )
        likelihood_precision = linalg.cho_solve(
            linalg.cho_factor(Sigma_likelihood), 
            np.eye(dim)
        )
        
        # Compute posterior precision and its Cholesky factor
        posterior_precision = prior_precision + likelihood_precision
        L_posterior = linalg.cholesky(posterior_precision, lower=True)
        Sigma_posterior_pred = linalg.cho_solve(
            (L_posterior, True), 
            np.eye(dim)
        )
        
    except np.linalg.LinAlgError:
        # Return large value if decomposition fails
        return 1e10

    # Transform mu_prior to the pose
    mu_prior = transform_mu(mu_likelihood, mu_prior_pose, version=2)

    # Compute predicted posterior parameters
    mu_posterior_pred = Sigma_posterior_pred @ (np.linalg.inv(Sigma_prior) @ mu_prior + np.linalg.inv(Sigma_likelihood) @ mu_likelihood)

    # Log-likelihood computation
    diff = obs - mu_posterior_pred  # Shape: (N, dim)
    
    # Use scipy's slogdet for numerical stability
    sign, logdet = np.linalg.slogdet(Sigma_posterior_pred)
    # if sign != 1:
    #     return 1e10  # Return large value if not positive definite
    
    # Compute quadratic terms efficiently
    inv_Sigma = np.linalg.inv(Sigma_posterior_pred)
    quad_terms = np.einsum('ij,ij->i', diff @ inv_Sigma, diff)  # Efficient diagonal of quadratic form
    
    # Compute log probabilities for each observation
    log_terms = -0.5 * dim * np.log(2 * np.pi) - 0.5 * logdet - 0.5 * quad_terms
    
    # Sum all log probabilities
    log_likelihood = np.sum(log_terms)

    # Posterior fitting loss
    loss_posterior = mu_loss(mu_posterior, mu_posterior_pred)

    # Modify the total loss to include regularization
    total_loss = -log_likelihood

    # Add small regularization term to ensure positive definiteness
    epsilon = 1e-6
    Sigma_prior += epsilon * np.eye(Sigma_prior.shape[0])
    Sigma_likelihood += epsilon * np.eye(Sigma_likelihood.shape[0])
    
    # At the end of the function, before returning:
    if iteration_count % 1000 == 0:  # Print every 10 iterations
        print(f"Iteration {iteration_count:4d} | "
              f"Total Loss: {total_loss:10.4f} | "
              f"Posterior Loss: {loss_posterior:10.4f} | "
              f"Trial Number: {trial_number}")
    
    return total_loss
