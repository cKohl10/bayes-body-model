from data_handler import Data2017
import numpy as np
import matplotlib.pyplot as plt

def compute_prior_covariance(Sigma_posterior, Sigma_likelihood, epsilon=1e-6):
    """
    Compute prior covariance matrix with numerical stability checks.
    
    Args:
        Sigma_posterior: Posterior covariance matrix
        Sigma_likelihood: Likelihood covariance matrix
        epsilon: Small regularization term
    """

    print(f"Sigma_posterior: {Sigma_posterior}")
    print(f"Sigma_likelihood: {Sigma_likelihood}")
    # Ensure matrices are symmetric
    Sigma_posterior = (Sigma_posterior + Sigma_posterior.T) / 2
    Sigma_likelihood = (Sigma_likelihood + Sigma_likelihood.T) / 2
    
    # Add small regularization to ensure matrices are positive definite
    reg_term = epsilon * np.eye(Sigma_posterior.shape[0])
    Sigma_posterior += reg_term
    Sigma_likelihood += reg_term
    
    # Compute inverse matrices
    Sigma_posterior_inv = np.linalg.inv(Sigma_posterior)
    Sigma_likelihood_inv = np.linalg.inv(Sigma_likelihood)
    
    # Compute prior inverse
    Sigma_prior_inv = Sigma_posterior_inv - Sigma_likelihood_inv
    
    # Add regularization if needed
    if not np.all(np.linalg.eigvals(Sigma_prior_inv) > 0):
        Sigma_prior_inv += reg_term
        
    # Compute prior covariance
    Sigma_prior = np.linalg.inv(Sigma_prior_inv)
    
    return Sigma_prior


if __name__ == "__main__":

    #### Hyperparameters ####
    # The likelihood covariance matrix
    Sigma_likelihood = np.array([[10.0, 0.0], [0.0, 10.0]])
    participants = [7,9]

    #########################

    data = Data2017() # Load in the dorsum data

    # We are focused on the combined data from test 1 and test 2
    for participant in participants:
        combined_data = data.dorsum_data[participant-1][2]
        combined_data['prior_cov'] = []

        # Get the prior covariance matrix using the following formula:
        # Sigma_prior = (Sigma_posterior - Sigma_likelihood)^-1
        for j in range(8):
            print(f"\nIndex {j}")
            Sigma_prior = compute_prior_covariance(combined_data['posterior_cov'][j], Sigma_likelihood)
            combined_data['prior_cov'].append(Sigma_prior)
            print(f"Sigma_prior: {Sigma_prior}")

            mew_posterior = np.array([combined_data['mean_judged']['X'][j]/combined_data['pix2cm'], combined_data['mean_judged']['Y'][j]/combined_data['pix2cm']])
            print(f"mew_posterior: {mew_posterior}")
            mew_likelihood = np.array([combined_data['pre_position']['X'][j]/combined_data['pix2cm'], combined_data['pre_position']['Y'][j]/combined_data['pix2cm']])
            print(f"mew_likelihood: {mew_likelihood}")
            mew_prior = Sigma_prior @ (np.linalg.inv(combined_data['posterior_cov'][j]) @ mew_posterior - np.linalg.inv(Sigma_likelihood) @ mew_likelihood)
            print(f"mew_prior: {mew_prior}")

        fig, ax = plt.subplots(figsize=(8, 6))
        data.plot_dorsum_data(combined_data, ax, 3, participant)
        data.plot_wrist_ellipses(combined_data['pre_position']['X']/combined_data['pix2cm'], combined_data['pre_position']['Y']/combined_data['pix2cm'], combined_data['prior_cov'], ax, 'b', 0.5)
        plt.show()


