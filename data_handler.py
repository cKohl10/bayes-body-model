# This script will import the xlsx data from the experiment and convert it into a pandas dataframe.

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np

class Data:
    def __init__(self):
        pass
    
    def import_data(self):
        pass
    
    def plot_data(self):
        pass    

    # Convert Excel column letters to numeric indices
    @staticmethod
    def excel_col_to_num(col_letter):
        """Convert Excel column letter to corresponding numeric index (0-based)"""
        num = 0
        for c in col_letter:
            num = num * 26 + (ord(c.upper()) - ord('A')) + 1
        return num - 1
    
    def plot_ellipse(self, mean, cov, ax, color='k', alpha=0.5):
        # Exit if the covariance matrix is not positive definite
        if np.any(np.isnan(cov)):
            print("Warning: Covariance matrix is not positive definite. Ellipse not plotted.")
            return

        # Plot an ellipse given a mean and covariance matrix
        try:
            eigenvalues, eigenvectors = np.linalg.eig(cov)
            
            # Check for negative eigenvalues
            if np.any(eigenvalues < 0):
                print("Warning: Negative eigenvalues found. Covariance matrix is not positive definite. Ellipse not plotted.")
                return
            
            angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
            width, height = 2 * np.sqrt(eigenvalues)
            ellipse = Ellipse(xy=mean, width=width, height=height, angle=angle, 
                         color=color, alpha=alpha)
            ax.add_patch(ellipse)
        except:
            print("Warning: Could not plot ellipse.")
            return
    
# Wrist data from Longo 2017
class Data2017(Data):
    def __init__(self):
        super().__init__()
        self.dorsum_data = self.import_data()
        
        # Connection key for the wrist outline
        self.connection_key = [[1, 2],
                               [1, 3],
                               [2, 4], 
                               [3, 4],
                               [3, 5],
                               [4, 6],
                               [5, 6],
                               [5, 7],
                               [6, 8],
                               [7, 8]]

    def import_data(self):
        
        ###### Data parameters (Excel sheet) ######
        dorsum_1_rows = [17, 24]
        dorsum_2_rows = [31, 38]

        pre_position_cols = [self.excel_col_to_num("R"), self.excel_col_to_num("S")]
        post_position_cols = [self.excel_col_to_num("T"), self.excel_col_to_num("U")]
        mean_position_cols = [self.excel_col_to_num("V"), self.excel_col_to_num("W")]

        indiv_judged_cols = [self.excel_col_to_num(col) for col in ["X", "Y", "Z", "AA", "AB", "AC"]]
        mean_judged_cols = [self.excel_col_to_num("AD"), self.excel_col_to_num("AE")]

        def extract_data(data, rows):
            row_slice = slice(rows[0]-1, rows[1])
            data_dict = {
                'pre_position': {
                    'X': data.iloc[row_slice, pre_position_cols[0]].astype(float).values,
                    'Y': data.iloc[row_slice, pre_position_cols[1]].astype(float).values,
                },
                'post_position': {
                    'X': data.iloc[row_slice, post_position_cols[0]].astype(float).values,
                    'Y': data.iloc[row_slice, post_position_cols[1]].astype(float).values,
                },
                'mean_position': {
                    'X': data.iloc[row_slice, mean_position_cols[0]].astype(float).values,
                    'Y': data.iloc[row_slice, mean_position_cols[1]].astype(float).values,
                },
                'indiv_judged':{
                    '1':{
                        'X': data.iloc[row_slice, indiv_judged_cols[0]].astype(float).values,
                        'Y': data.iloc[row_slice, indiv_judged_cols[1]].astype(float).values,
                    },
                    '2':{
                        'X': data.iloc[row_slice, indiv_judged_cols[2]].astype(float).values,
                        'Y': data.iloc[row_slice, indiv_judged_cols[3]].astype(float).values,
                    },
                    '3':{
                        'X': data.iloc[row_slice, indiv_judged_cols[4]].astype(float).values,
                        'Y': data.iloc[row_slice, indiv_judged_cols[5]].astype(float).values,
                    }
                },
                'mean_judged':{
                    'X': data.iloc[row_slice, mean_judged_cols[0]].astype(float).values,
                    'Y': data.iloc[row_slice, mean_judged_cols[1]].astype(float).values,
                },
                'pix2cm': float(data.iloc[11, 2])
            }

            data_dict['posterior_cov'] = self.covariance_matrix(data_dict)
            return data_dict
    
        dorsum_data = []
        for i in range(1, 13):
            data = pd.read_csv(f"data/Longo_2017/wrist_data_{i}.csv")
            data1 = extract_data(data, dorsum_1_rows)
            data2 = extract_data(data, dorsum_2_rows)
            data3 = self.combine_data(data1, data2)
            dorsum_data.append([data1, data2, data3])

        return dorsum_data
    
    def combine_data(self, data1, data2):
        data3 = {
            'pre_position': {'X': None, 'Y': None},
            'post_position': {'X': None, 'Y': None},
            'mean_position': {'X': None, 'Y': None},
            'mean_judged': {'X': None, 'Y': None},
            'prior_cov': [],
            'likelihood_cov': [],
            'total_prior_cov': [],
            'total_likelihood_cov': [],
            'prior_mean': {'X': [], 'Y': []},
            'indiv_judged': {
                '1': {'X': None, 'Y': None},
                '2': {'X': None, 'Y': None},
                '3': {'X': None, 'Y': None},
                '4': {'X': None, 'Y': None},
                '5': {'X': None, 'Y': None},
                '6': {'X': None, 'Y': None},
            },
            'pix2cm': None
        }
        # Stack arrays and take mean along first axis
        data3['pre_position']['X'] = (data1['pre_position']['X'] + data2['pre_position']['X']) / 2
        data3['pre_position']['Y'] = (data1['pre_position']['Y'] + data2['pre_position']['Y']) / 2
        data3['post_position']['X'] = (data1['post_position']['X'] + data2['post_position']['X']) / 2
        data3['post_position']['Y'] = (data1['post_position']['Y'] + data2['post_position']['Y']) / 2
        data3['mean_position']['X'] = (data1['mean_position']['X'] + data2['mean_position']['X']) / 2
        data3['mean_position']['Y'] = (data1['mean_position']['Y'] + data2['mean_position']['Y']) / 2
        data3['mean_judged']['X'] = (data1['mean_judged']['X'] + data2['mean_judged']['X']) / 2
        data3['mean_judged']['Y'] = (data1['mean_judged']['Y'] + data2['mean_judged']['Y']) / 2
        data3['indiv_judged']['1']['X'] = data1['indiv_judged']['1']['X']
        data3['indiv_judged']['1']['Y'] = data1['indiv_judged']['1']['Y']
        data3['indiv_judged']['2']['X'] = data1['indiv_judged']['2']['X']
        data3['indiv_judged']['2']['Y'] = data1['indiv_judged']['2']['Y']
        data3['indiv_judged']['3']['X'] = data1['indiv_judged']['3']['X']
        data3['indiv_judged']['3']['Y'] = data1['indiv_judged']['3']['Y']
        data3['indiv_judged']['4']['X'] = data2['indiv_judged']['1']['X']
        data3['indiv_judged']['4']['Y'] = data2['indiv_judged']['1']['Y']
        data3['indiv_judged']['5']['X'] = data2['indiv_judged']['2']['X']
        data3['indiv_judged']['5']['Y'] = data2['indiv_judged']['2']['Y']
        data3['indiv_judged']['6']['X'] = data2['indiv_judged']['3']['X']
        data3['indiv_judged']['6']['Y'] = data2['indiv_judged']['3']['Y']
        data3['pix2cm'] = (data1['pix2cm'] + data2['pix2cm']) / 2

        data3['posterior_cov'] = self.covariance_matrix(data3)
        data3['total_posterior_cov'] = self.total_covariance(data3)
        return data3  # Add return statement
    
    def covariance_matrix(self, data):
        covs = np.zeros((len(data['mean_judged']['X']), 2, 2))
        for j in range(len(data['mean_judged']['X'])):
            x_points = np.array([data['indiv_judged'][str(n+1)]['X'][j]/data['pix2cm'] for n in range(len(data['indiv_judged']))])
            y_points = np.array([data['indiv_judged'][str(n+1)]['Y'][j]/data['pix2cm'] for n in range(len(data['indiv_judged']))])
            points = np.vstack([x_points, y_points])
            cov = np.cov(points)
            covs[j] = cov
        return covs
        
    def total_covariance(self, data):
        # Create a 16x1 array of all points [x1:8, y1:8]
        points = []
        for j in range(len(data['mean_judged']['X'])):  # For each landmark point
            # Add all x coordinates for this landmark
            points.append([data['indiv_judged'][str(n+1)]['X'][j]/data['pix2cm'] for n in range(len(data['indiv_judged']))])
            # Add all y coordinates for this landmark
            points.append([data['indiv_judged'][str(n+1)]['Y'][j]/data['pix2cm'] for n in range(len(data['indiv_judged']))])
        
        points = np.array(points)  # Convert to numpy array
        cov = np.cov(points)  # Calculate 16x16 covariance matrix
        print(f"cov: {cov}")
        return cov

    def plot_data(self, participant_num=None):
        # This plot will plot the pre positions of the dorsum, the individual judged positions, and the mean judged position
        # If a participant number is provided, it will only plot the data for that participant

        if participant_num is not None:
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12,6))
            dorsum_data = self.dorsum_data[participant_num-1]
            self.plot_dorsum_data(dorsum_data[0], ax1, 1, participant_num)
            self.plot_dorsum_data(dorsum_data[1], ax2, 2, participant_num)
            self.plot_dorsum_data(dorsum_data[2], ax3, 3, participant_num)
            fig.show()

        else:
            for i in range(len(self.dorsum_data)):
                fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12,6))
                dorsum_data = self.dorsum_data[i]
                self.plot_dorsum_data(dorsum_data[0], ax1, 1, i+1)
                self.plot_dorsum_data(dorsum_data[1], ax2, 2, i+1)
                self.plot_dorsum_data(dorsum_data[2], ax3, 3, i+1)
                fig.show()

        plt.show()

    def plot_connection(self, ax, data_segment, pix2cm, color, label=None):
        for i, connection in enumerate(self.connection_key):
            if i == 0:
                ax.plot([data_segment['X'][connection[0]-1]/pix2cm, data_segment['X'][connection[1]-1]/pix2cm], [data_segment['Y'][connection[0]-1]/pix2cm, data_segment['Y'][connection[1]-1]/pix2cm], color, label=label)
            else:
                ax.plot([data_segment['X'][connection[0]-1]/pix2cm, data_segment['X'][connection[1]-1]/pix2cm], [data_segment['Y'][connection[0]-1]/pix2cm, data_segment['Y'][connection[1]-1]/pix2cm], color)

    def plot_dorsum_data(self, dorsum_data, ax, test_num, participant):

        # Plot the pre positions of the dorsum
        #ax.scatter(dorsum_data['pre_position']['X']/dorsum_data['pix2cm'], dorsum_data['pre_position']['Y']/dorsum_data['pix2cm'], label='Prior')
        self.plot_connection(ax, dorsum_data['pre_position'], dorsum_data['pix2cm'], 'b-', label='Likelihood')

        # self.plot_wrist_ellipses(dorsum_data['mean_judged']['X']/dorsum_data['pix2cm'], dorsum_data['mean_judged']['Y']/dorsum_data['pix2cm'], dorsum_data['posterior_cov'], ax, 'r', 0.5)

        # Plot the mean judged position
        #ax.scatter(dorsum_data['mean_judged']['X']/dorsum_data['pix2cm'], dorsum_data['mean_judged']['Y']/dorsum_data['pix2cm'], label='Mean Posterior', color='k')
        self.plot_connection(ax, dorsum_data['mean_judged'], dorsum_data['pix2cm'], 'k-', label='Posterior')
        ax.legend()
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel('X (cm)')
        ax.set_ylabel('Y (cm)')
        if test_num == 3:
            ax.set_title(f'Participant {participant}')
        else:
            ax.set_title(f'Participant {participant} Test {test_num}')

    def plot_wrist_ellipses(self, mean_x_data, mean_y_data, cov_data, ax, color='k', alpha=0.5):
        # The data needs to be a vertical stack of the x and y positions
        # There are 3 judgements for each test and two tests per participant
        # It is assumed that the data is only for one participant
        means = np.vstack([mean_x_data, mean_y_data])
        for j in range(len(means[0])):
            self.plot_ellipse(np.array([means[0,j], means[1,j]]), cov_data[j], ax, color, alpha)

# Palm data from Longo 2016
class Data2016(Data):
    def __init__(self):
        super().__init__()
        self.data = self.import_data()
        self.keys = ["small", "medium", "large"]
        self.directions = ["along", "across"]
        self.inference = ["judged", "actual"]

    # Import the data from the xlsx file
    def import_data(self):

        ###### Data parameters (Excel sheet) ######
        # Small Distances:
        small_along_judged_rows = [3,14]
        small_across_judged_rows = [17,28]
        small_along_actual_rows = [31,42]
        small_across_actual_rows = [45,56]

        # Medium Distances:
        medium_along_judged_rows = [61,68]
        medium_across_judged_rows = [71,78]
        medium_along_actual_rows = [82,89]
        medium_across_actual_rows = [92,99]

        # Large Distances:
        large_along_judged_rows = [104,107]
        large_across_judged_rows = [110,113]
        large_along_actual_rows = [116,119]
        large_across_actual_rows = [122,125]

        # Column definitions
        mean_col = self.excel_col_to_num("D")  # 3
        std_col = self.excel_col_to_num("E")   # 4
        sample_start = self.excel_col_to_num("I")    # 8
        sample_end = self.excel_col_to_num("AK")     # 36
        sample_cols = slice(sample_start, sample_end + 1)  # Include the end column

        data = pd.read_csv("data/Longo_2016/all2.csv")
        
        def extract_data(rows):
            # Helper function to extract different data types for a given row range
            row_slice = slice(rows[0]-1, rows[1])
            return {
                'mean': data.iloc[row_slice, mean_col].values,
                'std': data.iloc[row_slice, std_col].values,
                'samples': data.iloc[row_slice, sample_cols].values
            }

        formatted_data = {
            'small': {
                'along': {
                    'judged': extract_data(small_along_judged_rows),
                    'actual': extract_data(small_along_actual_rows)
                },
                'across': {
                    'judged': extract_data(small_across_judged_rows),
                    'actual': extract_data(small_across_actual_rows)
                }
            },
            'medium': {
                'along': {
                    'judged': extract_data(medium_along_judged_rows),
                    'actual': extract_data(medium_along_actual_rows)
                },
                'across': {
                    'judged': extract_data(medium_across_judged_rows),
                    'actual': extract_data(medium_across_actual_rows)
                }
            },
            'large': {
                'along': {
                    'judged': extract_data(large_along_judged_rows),
                    'actual': extract_data(large_along_actual_rows)
                },
                'across': {
                    'judged': extract_data(large_across_judged_rows),
                    'actual': extract_data(large_across_actual_rows)
                }
            }
        }

        return formatted_data
    
    def plot_data(self):
        # # Plot the data points for each distance and direction
        # for key in self.keys:
        #     for direction in self.directions:
        #         for inference in self.inference:
                    
        # plt.legend()
        # plt.show()

        pass

if __name__ == "__main__":
    data = Data2017()
    data.plot_data(participant_num=None)
