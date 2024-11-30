from data_handler import Data2017
from lib.optimizers import double_objective, loc_objective, transform_mu, iteration_count
import lib.optimizers as opt
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

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
    #Sigma_likelihood = np.array([[10.0, 0.0], [0.0, 10.0]])
    #participants = [7,9]
    participants = list(range(9, 12))

    #########################

    data = Data2017() # Load in the dorsum data

    # We are focused on the combined data from test 1 and test 2
    for participant in participants:
        fig_list = []
        loss_values = []
        for k in range(20):
            # Optimization
            try:
                combined_data = data.dorsum_data[participant-1][2].copy()
                # Analytical approach at computing the prior covariance matrix
                # Get the prior covariance matrix using the following formula:
                # Sigma_prior = (Sigma_posterior - Sigma_likelihood)^-1
                # for j in range(8):
                #     print(f"\nIndex {j}")
                #     Sigma_prior = compute_prior_covariance(combined_data['posterior_cov'][j], Sigma_likelihood)
                #     combined_data['prior_cov'].append(Sigma_prior)
                #     print(f"Sigma_prior: {Sigma_prior}")

                #     mew_posterior = np.array([combined_data['mean_judged']['X'][j]/combined_data['pix2cm'], combined_data['mean_judged']['Y'][j]/combined_data['pix2cm']])
                #     print(f"mew_posterior: {mew_posterior}")
                #     mew_likelihood = np.array([combined_data['pre_position']['X'][j]/combined_data['pix2cm'], combined_data['pre_position']['Y'][j]/combined_data['pix2cm']])
                #     print(f"mew_likelihood: {mew_likelihood}")
                #     mew_prior = Sigma_prior @ (np.linalg.inv(combined_data['posterior_cov'][j]) @ mew_posterior - np.linalg.inv(Sigma_likelihood) @ mew_likelihood)
                #     print(f"mew_prior: {mew_prior}")

                # Optimized approach at computing the prior covariance matrix
                # We are optimizing the following:
                # min -log(likelihood) + loss_mean + loss_cov
                # where loss_mean = ||mu_posterior - mu_posterior_pred||^2
                # and loss_cov = ||Sigma_posterior - Sigma_posterior_pred||^2
                # Observations
                obs = np.array([np.concatenate((combined_data['indiv_judged'][str(i+1)]['X']/combined_data['pix2cm'], combined_data['indiv_judged'][str(i+1)]['Y']/combined_data['pix2cm'])) for i in range(len(combined_data['indiv_judged']))])
                mu_likelihood = np.concatenate((combined_data['pre_position']['X']/combined_data['pix2cm'], combined_data['pre_position']['Y']/combined_data['pix2cm']))
                mu_posterior = np.concatenate((combined_data['mean_judged']['X']/combined_data['pix2cm'], combined_data['mean_judged']['Y']/combined_data['pix2cm']))
                Sigma_posterior = combined_data['total_posterior_cov']

                print(f"obs: {obs}")
                print(f"mu_likelihood: {mu_likelihood}")
                print(f"mu_posterior: {mu_posterior}")
                print(f"Sigma_posterior: {Sigma_posterior}")

                # Dimensions
                dim = 16 # (x,y) * 8
                pose_dim = 3

                # Covariance bounds
                cov_max = 100
                cov_min = 1e-2

                # Initial guesses
                mu_prior_init = np.zeros(pose_dim)  # Initial guess for prior mean pose
                mu_prior_init[0] = np.random.uniform(15.0, 25.0)
                mu_prior_init[1] = np.random.uniform(15.0, 25.0)
                mu_prior_init[2] = np.random.uniform(-np.pi*(1/16), np.pi*(1/16))
                # Sigma_prior_init = np.eye(dim)  # Initial guess for prior covariance
                # Sigma_likelihood_init = np.eye(dim)  # Assume diagonal for simplicity
                Sigma_prior_init = np.random.uniform(cov_min, cov_max, dim)
                Sigma_likelihood_init = np.random.uniform(cov_min, cov_max, dim)

                # Flatten initial parameters
                init_params = np.hstack([
                    mu_prior_init, 
                    Sigma_prior_init, 
                    Sigma_likelihood_init
                ])

                # Reset the counter in optimizers.py
                opt.iteration_count = 0


                result = minimize(
                    loc_objective, 
                init_params, 
                args=(obs, mu_likelihood, mu_posterior, Sigma_posterior, k+1),
                method='L-BFGS-B',
                options={
                    'maxiter': 500000,  # Maximum number of iterations
                    'maxfun': 500000,   # Maximum number of function evaluations
                    'ftol': 1e-9,        # Function tolerance (optional)
                    'gtol': 1e-9         # Gradient tolerance (optional)
                    }
                )  
                loss_values.append(result.fun)
                print(f"Optimization loss value: {result.fun}")

            except Exception as e:
                print(f"Failed to optimize for participant {participant}: {e}")
                continue

            # Extract results
            optimized_params = result.x
            mu_prior_pose_opt = optimized_params[:pose_dim]
            print(f"mu_prior_pose_opt: {mu_prior_pose_opt}")
            mu_prior_opt = transform_mu(mu_likelihood, mu_prior_pose_opt)
            # Sigma_prior_opt = optimized_params[pose_dim:pose_dim + dim**2].reshape(dim, dim)
            # Sigma_prior_opt = np.dot(Sigma_prior_opt, Sigma_prior_opt.T)  # Ensure positive definite
            # Sigma_likelihood_opt = optimized_params[pose_dim + dim**2:].reshape(dim, dim)
            # Sigma_likelihood_opt = np.dot(Sigma_likelihood_opt, Sigma_likelihood_opt.T)

            # Version where the free paramters of the covariance matrix are only the diagonals
            Sigma_prior_opt = np.diag(optimized_params[pose_dim:pose_dim + dim])
            Sigma_likelihood_opt = np.diag(optimized_params[pose_dim + dim:])

            for j in range(8):
                # Extract 2x2 covariance matrices for x,y coordinates at position j
                x_y_indices = [j, j+8]  # j for x, j+8 for y coordinate
                prior_cov_2x2 = Sigma_prior_opt[np.ix_(x_y_indices, x_y_indices)]
                likelihood_cov_2x2 = Sigma_likelihood_opt[np.ix_(x_y_indices, x_y_indices)]
                
                combined_data['prior_cov'].append(prior_cov_2x2)
                combined_data['likelihood_cov'].append(likelihood_cov_2x2)
                combined_data['prior_mean']['X'].append(mu_prior_opt[j])
                combined_data['prior_mean']['Y'].append(mu_prior_opt[j+8])

            x_points = np.array(combined_data['prior_mean']['X'])
            y_points = np.array(combined_data['prior_mean']['Y'])

            best_prior_mean = {'X': x_points, 'Y': y_points}


            # Display results
            # print("Optimized Prior Mean:", mu_prior_opt)
            # print("Optimized Prior Covariance:\n", Sigma_prior_opt)
            # print("Optimized Likelihood Covariance:\n", Sigma_likelihood_opt)
            # print(f"combined_data['prior_cov']: {combined_data['prior_cov']}")
            # print(f"combined_data['likelihood_cov']: {combined_data['likelihood_cov']}")
            # print(f"combined_data['prior_mean']: {combined_data['prior_mean']}")

            # Initial Pose
            initial_pose = transform_mu(mu_likelihood, mu_prior_init)
            initial_p = {'X': initial_pose[:8], 'Y': initial_pose[8:]}
                

            fig, ax = plt.subplots(figsize=(8, 6))
            data.plot_dorsum_data(combined_data, ax, 3, participant)
            data.plot_wrist_ellipses(combined_data['pre_position']['X']/combined_data['pix2cm'], combined_data['pre_position']['Y']/combined_data['pix2cm'], combined_data['likelihood_cov'], ax, 'b', 0.1)
            data.plot_wrist_ellipses(combined_data['prior_mean']['X'], combined_data['prior_mean']['Y'], combined_data['prior_cov'], ax, 'r', 0.1)
            # plt.show()

            # fig, ax = plt.subplots(figsize=(8, 6))
            data.plot_connection(ax, best_prior_mean,1, 'g', label='Fitted Prior')
            # data.plot_connection(ax, initial_p, 1, 'r', label='Optimizer Initial')
            plt.title(f"Participant {participant}, Trial {k+1}, Loss: {loss_values[k]:.2f}")
            plt.legend()
            plt.show()
            #fig_list.append(fig)

        # Get the lowest loss value index
        # lowest_loss_index = np.argmin(loss_values)
        # fig_list[lowest_loss_index].savefig(f'figures/fitted_prior_{participant}.png')

