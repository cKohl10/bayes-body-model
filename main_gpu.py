import torch
import torch.nn as nn
import torch.optim as optim
import torch.cuda as cuda
import numpy as np
import matplotlib.pyplot as plt

import lib.optimizers_gpu as opt_gpu
import lib.optimizers as opt
from data_handler import Data2017

if __name__ == "__main__":
    device = torch.device("cuda" if cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    participants = list(range(1, 12))
    diag_only = False

    data = Data2017()

    for participant in participants:
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

        # Optimization
        bayes_net = opt_gpu.BayesFittingNet(obs, mu_likelihood, mu_posterior, device, diag_only=diag_only)

        # Use the ADAM optimizer
        optimizer = optim.Adam(bayes_net.parameters(), lr=0.01)

        # Training loop
        n_epochs = 10000
        best_loss = float('inf')
        best_state = None

        for epoch in range(n_epochs):
            optimizer.zero_grad()
            loss = bayes_net()
            
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_state = {
                    'mu_prior_pose': bayes_net.mu_prior_pose.detach().clone(),
                    'Sigma_prior': bayes_net.construct_covariance_matrices()[0].detach().clone(),
                    'Sigma_likelihood': bayes_net.construct_covariance_matrices()[1].detach().clone()
                }
            
            loss.backward()
            optimizer.step()
            
            # Early stopping condition
            if loss.item() < 1e-6:
                break

        # Extract results from the best state (not directly from optimized_params)
        mu_prior_pose_opt = best_state['mu_prior_pose'].cpu().numpy()
        mu_prior_opt = opt.transform_mu(mu_likelihood, mu_prior_pose_opt, version=2)
        mu_prior_opt = opt.mu_reorganize(mu_prior_opt, 2)

        # Get covariance matrices from best state
        Sigma_prior_opt = best_state['Sigma_prior'].cpu().numpy()
        Sigma_likelihood_opt = best_state['Sigma_likelihood'].cpu().numpy()

        # print(f"Sigma_prior_opt shape: {Sigma_prior_opt.shape}")
        # print(f"Sigma_likelihood_opt shape: {Sigma_likelihood_opt.shape}")

        for j in range(8):
            prior_cov_2x2 = Sigma_prior_opt[j*2:(j+1)*2, j*2:(j+1)*2]
            likelihood_cov_2x2 = Sigma_likelihood_opt[j*2:(j+1)*2, j*2:(j+1)*2]

            # print(f"prior_cov_2x2 shape: {prior_cov_2x2.shape}")
            # print(f"likelihood_cov_2x2 shape: {likelihood_cov_2x2.shape}")
            
            combined_data['prior_cov'].append(prior_cov_2x2)
            combined_data['likelihood_cov'].append(likelihood_cov_2x2)
            combined_data['prior_mean']['X'].append(mu_prior_opt[j])
            combined_data['prior_mean']['Y'].append(mu_prior_opt[j+8])

        x_points = np.array(combined_data['prior_mean']['X'])
        y_points = np.array(combined_data['prior_mean']['Y'])

        best_prior_mean = {'X': x_points, 'Y': y_points}

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
        fig.suptitle(f"Participant {participant}\nFinal Loss: {best_loss:.2f}", 
                        fontsize='large', 
                        y=1.02)

        # Add legend to the first subplot
        axs[0].legend()
        axs[1].legend()
        axs[2].legend()

        plt.show()


