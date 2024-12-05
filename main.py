from data_handler import Data2017
from lib.optimizers import general_objective, transform_mu, iteration_count
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
    participants = list(range(1, 12))
    diag_only = True

    #########################

    data = Data2017() # Load in the dorsum data

    # We are focused on the combined data from test 1 and test 2
    for participant in participants:
        fig_list = []
        loss_values = []

        combined_data = data.dorsum_data[participant-1][2].copy()

        # Observations
        obs = np.zeros((6, 16))
        for i in range(len(combined_data['indiv_judged'])):
            obs_temp = np.concatenate((combined_data['indiv_judged'][str(i+1)]['X']/combined_data['pix2cm'], combined_data['indiv_judged'][str(i+1)]['Y']/combined_data['pix2cm']))
            # print(f"obs_temp Before: {obs_temp}")
            obs_temp = opt.mu_reorganize(obs_temp, 1)
            # print(f"obs_temp After: {obs_temp}")
            obs[i, :] = obs_temp
        #obs = np.array([np.concatenate((combined_data['indiv_judged'][str(i+1)]['X']/combined_data['pix2cm'], combined_data['indiv_judged'][str(i+1)]['Y']/combined_data['pix2cm'])) for i in range(len(combined_data['indiv_judged']))])
        mu_likelihood = np.concatenate((combined_data['pre_position']['X']/combined_data['pix2cm'], combined_data['pre_position']['Y']/combined_data['pix2cm']))
        mu_posterior = np.concatenate((combined_data['mean_judged']['X']/combined_data['pix2cm'], combined_data['mean_judged']['Y']/combined_data['pix2cm']))
        Sigma_posterior = combined_data['total_posterior_cov']

        mu_likelihood = opt.mu_reorganize(mu_likelihood, 1)
        mu_posterior = opt.mu_reorganize(mu_posterior, 1)

        # Check if any of the covariance matrices are not positive definite
        # if not np.all(np.linalg.eigvals(Sigma_posterior) > 0):
        #     print(f"Sigma_posterior is not positive definite for participant {participant}, trial {k+1}")
        #     continue


        # Dimensions
        if diag_only:
            dim = 16 # (x,y) * 8
        else:
            dim = 32 # (x,y) * 8
        
        pose_dim = 3

        # Covariance bounds
        cov_max = 100
        cov_min = 1e-2

        for k in range(10):
            try:
                # Optimization
                combined_data['prior_cov'] = []
                combined_data['likelihood_cov'] = []
                combined_data['prior_mean'] = {'X': [], 'Y': []}

                # if opt.check_cov_for_nan(combined_data['']):
                #     print(f"Warning: Covariance matrix contains NaNs for participant {participant}, trial {k+1}")
                #     continue

                # Initial guesses
                mu_prior_init = np.zeros(pose_dim)  # Initial guess for prior mean pose
                mu_prior_init[0] = np.random.uniform(15.0, 25.0)
                mu_prior_init[1] = np.random.uniform(15.0, 25.0)
                mu_prior_init[2] = np.random.uniform(-np.pi*(1/16), np.pi*(1/16))
                Sigma_prior_init = np.random.uniform(cov_min, cov_max, dim)
                Sigma_likelihood_init = np.random.uniform(cov_min, cov_max, dim)

                if not diag_only:   
                    for i in range(8):
                        # make sure the covariances are not more than the variance
                        b = i*4
                        Sigma_prior_init[b+1] = np.minimum(Sigma_prior_init[b], Sigma_prior_init[b+3])/2
                        Sigma_prior_init[b+2] = np.minimum(Sigma_prior_init[b], Sigma_prior_init[b+3])/2
                        Sigma_likelihood_init[b+1] = np.minimum(Sigma_likelihood_init[b], Sigma_likelihood_init[b+3])/2
                        Sigma_likelihood_init[b+2] = np.minimum(Sigma_likelihood_init[b], Sigma_likelihood_init[b+3])/2

                # Flatten initial parameters
                init_params = np.hstack([
                    mu_prior_init, 
                    Sigma_prior_init, 
                    Sigma_likelihood_init
                ])

                # Reset the counter in optimizers.py
                opt.iteration_count = 0

                print(f"Beginning optimization for participant {participant}, trial {k+1}")

                result = minimize(
                    general_objective, 
                init_params, 
                args=(obs, mu_likelihood, mu_posterior, k+1, diag_only),
                method='L-BFGS-B',
                options={
                    'maxiter': 1000000,  # Maximum number of iterations
                    'maxfun': 1000000,   # Maximum number of function evaluations
                    'ftol': 1e-8,        # Function tolerance (optional)
                    'gtol': 1e-8         # Gradient tolerance (optional)
                    }
                )  
                loss_values.append(result.fun)
                print(f"Optimization loss value: {loss_values[k]}")

                # except Exception as e:
                #     print(f"Failed to optimize for participant {participant}: {e}")
                #     continue

                # Extract results
                optimized_params = result.x
                mu_prior_pose_opt = optimized_params[:pose_dim]
                print(f"mu_prior_pose_opt: {mu_prior_pose_opt}")
                mu_prior_opt = transform_mu(mu_likelihood, mu_prior_pose_opt, version=2)
                mu_prior_opt = opt.mu_reorganize(mu_prior_opt, 2)

                # Version where the free paramters of the covariance matrix are only the diagonals
                Sigma_prior_opt = optimized_params[pose_dim:pose_dim + dim]
                Sigma_likelihood_opt = optimized_params[pose_dim + dim:]

                for j in range(8):
                    # Extract 2x2 covariance matrices for x,y coordinates at position j
                    # x_y_indices = [j, j+8]  # j for x, j+8 for y coordinate
                    # prior_cov_2x2 = Sigma_prior_opt[np.ix_(x_y_indices, x_y_indices)]
                    # likelihood_cov_2x2 = Sigma_likelihood_opt[np.ix_(x_y_indices, x_y_indices)]
                    if diag_only:
                        prior_cov_2x2 = np.diag(Sigma_prior_opt[j*2:(j+1)*2])
                        likelihood_cov_2x2 = np.diag(Sigma_likelihood_opt[j*2:(j+1)*2])
                    else:
                        prior_cov_2x2, likelihood_cov_2x2 = opt.extract_individual_covariance(Sigma_prior_opt, Sigma_likelihood_opt, j)
                        print(f"prior_cov_2x2: {prior_cov_2x2}")
                    
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
                # initial_p = {'X': initial_pose[:8], 'Y': initial_pose[8:]}
                    

                fig, axs = plt.subplots(1, 3, figsize=(18, 6))  # Create a figure with 3 side-by-side subplots

                # Plot prior mean on the second subplot
                data.plot_dorsum_data(combined_data, axs[0], 3, participant)
                data.plot_wrist_ellipses(combined_data['prior_mean']['X'], combined_data['prior_mean']['Y'], combined_data['prior_cov'], axs[0], 'g', 0.1)
                data.plot_connection(axs[0], best_prior_mean, 1, 'g', label='Fitted Prior')
                axs[0].set_title('Prior Mean')

                # Plot mean judged on the third subplot
                data.plot_dorsum_data(combined_data, axs[1], 3, participant)
                data.plot_connection(axs[1], best_prior_mean, 1, 'g', label='Fitted Prior')
                data.plot_wrist_ellipses(combined_data['mean_judged']['X']/combined_data['pix2cm'], combined_data['mean_judged']['Y']/combined_data['pix2cm'], combined_data['posterior_cov'], axs[1], 'k', 0.1)
                axs[1].set_title('Mean Judged')

                # Plot dorsum data on the first subplot
                data.plot_dorsum_data(combined_data, axs[2], 3, participant)
                data.plot_connection(axs[2], best_prior_mean, 1, 'g', label='Fitted Prior')
                data.plot_wrist_ellipses(combined_data['pre_position']['X']/combined_data['pix2cm'], combined_data['pre_position']['Y']/combined_data['pix2cm'], combined_data['likelihood_cov'], axs[2], 'b', 0.1)
                axs[2].set_title('Pre Position')


                # Set a common title for the entire figure
                fig.suptitle(f"Participant {participant}\nTrial {k+1}, Loss: {loss_values[k]:.2f}", 
                            fontsize='large', 
                            y=1.02)

                # Add legend to the first subplot
                axs[0].legend()
                axs[1].legend()
                axs[2].legend()

                # plt.show()
                fig_list.append(fig)
            except Exception as e:
                print(f"Failed to optimize for participant {participant}: {e}")
                continue

        # Get the lowest loss value index
        if len(loss_values) > 0:
            lowest_loss_index = np.argmin(loss_values)
            fig_list[lowest_loss_index].savefig(f'figures/fitted_prior_{participant}_diag={diag_only}.png')

