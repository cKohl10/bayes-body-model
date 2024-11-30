import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy import linalg
# Add at the top of the file, after imports
iteration_count = 0

def transform_mu(mu, pose):
    # pose is in the form [x, y, theta]
    # mu is in the form [x1, x2, ..., x8, y1, y2, ..., y8]
    
    # Reshape points into 2xN array for easier manipulation
    points = np.vstack([mu[:8], mu[8:]])  # Shape: (2, 8)
    
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
    
    # Return in original format [x1, ..., x8, y1, ..., y8]
    return np.concatenate((points[0], points[1]))

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

def loc_objective(params, obs, mu_likelihood, mu_posterior, Sigma_posterior, trial_number):
    # This optimization can only adjust the x,y,theta of the prior, keeping the original shape from the likelihood.

    # The params variable is structured as follows:
    # params = [x, y, theta, sigma_prior_1, sigma_prior_2, ..., sigma_prior_dim, sigma_likelihood_1, sigma_likelihood_2, ..., sigma_likelihood_dim]
    # where:
    # - x, y, theta are the pose parameters (3 elements)
    # - sigma_prior_1, sigma_prior_2, ..., sigma_prior_dim are the diagonal elements of the prior covariance matrix (dim elements)
    # - sigma_likelihood_1, sigma_likelihood_2, ..., sigma_likelihood_dim are the diagonal elements of the likelihood covariance matrix (dim elements)
    # 
    # Visual depiction:
    # params = [x, y, theta, 
    #           sigma_prior_1, sigma_prior_2, ..., sigma_prior_dim, 
    #           sigma_likelihood_1, sigma_likelihood_2, ..., sigma_likelihood_dim]
    # 
    # Example for dim=2:
    # params = [x, y, theta, 
    #           sigma_prior_1, sigma_prior_2, 
    #           sigma_likelihood_1, sigma_likelihood_2]

    # Add global counter
    global iteration_count
    iteration_count += 1

    # Dimensions
    N, dim = obs.shape

    # Unpack params and create positive definite matrices using Cholesky
    mu_prior_pose = params[:3]
    try:
        # L_prior = params[3:3 + dim**2].reshape(dim, dim)
        # L_likelihood = params[3 + dim**2:].reshape(dim, dim)

        Sigma_prior = np.diag(params[3:3 + dim])
        Sigma_likelihood = np.diag(params[3 + dim:])
        
        # Create positive definite matrices
        # Sigma_prior = L_prior @ L_prior.T
        # Sigma_likelihood = L_likelihood @ L_likelihood.T
        
        # Add regularization coefficient
        cov_reg_strength = 0.0  # Increase this value to make covariance changes more costly
        
        # Add regularization loss for covariance matrices
        cov_reg_loss = (np.linalg.norm(Sigma_prior - np.eye(dim)) + 
                       np.linalg.norm(Sigma_likelihood - np.eye(dim))) * cov_reg_strength
        
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
    mu_prior = transform_mu(mu_likelihood, mu_prior_pose)

    # Compute predicted posterior parameters
    mu_posterior_pred = Sigma_posterior_pred @ (np.linalg.inv(Sigma_prior) @ mu_prior + np.linalg.inv(Sigma_likelihood) @ mu_likelihood)

    # Log-likelihood computation
    diff = obs - mu_posterior_pred  # Shape: (N, dim)
    
    # Use scipy's slogdet for numerical stability
    sign, logdet = np.linalg.slogdet(Sigma_posterior_pred)
    if sign != 1:
        return 1e10  # Return large value if not positive definite
    
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
    total_loss = -log_likelihood + cov_reg_loss

    # At the end of the function, before returning:
    if iteration_count % 100 == 0:  # Print every 10 iterations
        print(f"Iteration {iteration_count:4d} | "
              f"Total Loss: {total_loss:10.4f} | "
              f"Posterior Loss: {loss_posterior:10.4f} | "
              f"Log-Likelihood: {log_likelihood:10.4f} | "
              f"Trial Number: {trial_number}")
    
    return total_loss


# Combined objective function
def double_objective(params, connection_key, obs, mu_likelihood, mu_posterior, Sigma_posterior):
    # Add global counter
    global iteration_count
    iteration_count += 1

    # Dimensions
    N, dim = obs.shape

    # Unpack parameters
    mu_prior = params[:dim]
    Sigma_prior_flat = params[dim:dim + dim**2]
    Sigma_prior = Sigma_prior_flat.reshape(dim, dim)
    Sigma_prior = Sigma_prior @ Sigma_prior.T + 1e-6 * np.eye(dim)  # Ensure positive definite

    Sigma_likelihood_flat = params[dim + dim**2:]
    Sigma_likelihood = Sigma_likelihood_flat.reshape(dim, dim)
    Sigma_likelihood = Sigma_likelihood @ Sigma_likelihood.T + 1e-6 * np.eye(dim)  # Ensure positive definite

    # Compute predicted posterior parameters
    Sigma_posterior_pred = np.linalg.inv(np.linalg.inv(Sigma_prior) + np.linalg.inv(Sigma_likelihood))
    mu_posterior_pred = Sigma_posterior_pred @ (np.linalg.inv(Sigma_prior) @ mu_prior + np.linalg.inv(Sigma_likelihood) @ mu_likelihood)

    # Log-likelihood computation
    diff = obs - mu_posterior_pred  # Shape: (N, dim)

    # Use scipy's slogdet for numerical stability
    sign, logdet = np.linalg.slogdet(Sigma_posterior_pred)
    if sign != 1:
        raise ValueError("Covariance matrix is not positive definite.")
    
    # Compute quadratic terms efficiently
    inv_Sigma = np.linalg.inv(Sigma_posterior_pred)
    quad_terms = np.einsum('ij,ij->i', diff @ inv_Sigma, diff)  # Efficient diagonal of quadratic form

    # Compute log probabilities
    log_terms = -0.5 * dim * np.log(2 * np.pi) - 0.5 * logdet - 0.5 * quad_terms

    # Total log-likelihood
    log_likelihood = np.sum(log_terms)

    # Penalty terms (if any)
    loss_mean = np.linalg.norm(mu_posterior - mu_posterior_pred)**2
    loss_cov = np.linalg.norm(Sigma_posterior - Sigma_posterior_pred)**2
    loss_prior = prior_objective(connection_key, mu_prior, mu_likelihood)

    # Combine loss terms
    total_loss = loss_prior

    # At the end of the function, before returning:
    if iteration_count % 100 == 0:  # Print every 10 iterations
        print(f"Iteration {iteration_count:4d} | "
              f"Total Loss: {total_loss:10.4f} | "
              f"Log-Likelihood: {-log_likelihood:10.4f} | "
              f"Mean Loss: {loss_mean:10.4f} | "
              f"Cov Loss: {loss_cov:10.4f}")
    
    return total_loss

if __name__ == "__main__":
    # Given data (example values)
    obs = np.array([[1.4, 2.4], [1.6, 2.6], [1.5, 2.5]])  # Observations (N samples, dim features)
    mu_likelihood = np.array([1.0, 2.0])  # Likelihood mean
    mu_posterior = np.array([1.5, 2.5])  # Posterior mean
    Sigma_posterior = np.array([[0.2, 0.1], [0.1, 0.3]])  # Posterior covariance

    # Dimensions
    N, dim = obs.shape

    # Initial guesses
    mu_prior_init = np.copy(mu_posterior)  # Initial guess for prior mean
    Sigma_prior_init = np.copy(Sigma_posterior)  # Initial guess for prior covariance
    Sigma_likelihood_init = np.copy(Sigma_posterior)  # Assume diagonal for simplicity

    # Flatten initial parameters
    init_params = np.hstack([
        mu_prior_init, 
        Sigma_prior_init.flatten(), 
        Sigma_likelihood_init.flatten()
    ])

    # Optimization
    result = minimize(
        double_objective, 
        init_params, 
        args=(obs, mu_likelihood, mu_posterior, Sigma_posterior),
        method='L-BFGS-B',
        options={'disp': True}  # Add this line to show optimization details
    )

    # Extract results
    optimized_params = result.x
    mu_prior_opt = optimized_params[:dim]
    Sigma_prior_opt = optimized_params[dim:dim + dim**2].reshape(dim, dim)
    Sigma_prior_opt = np.dot(Sigma_prior_opt, Sigma_prior_opt.T)  # Ensure positive definite
    Sigma_likelihood_opt = optimized_params[dim + dim**2:].reshape(dim, dim)
    Sigma_likelihood_opt = np.dot(Sigma_likelihood_opt, Sigma_likelihood_opt.T)

    # Display results
    print("Optimized Prior Mean:", mu_prior_opt)
    print("Optimized Prior Covariance:\n", Sigma_prior_opt)
    print("Optimized Likelihood Covariance:\n", Sigma_likelihood_opt)
