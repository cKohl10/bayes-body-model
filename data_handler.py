# This script will import the xlsx data from the experiment and convert it into a pandas dataframe.

import pandas as pd
import matplotlib.pyplot as plt

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
            return {
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
    
        dorsum_data = []
        for i in range(1, 12):
            data = pd.read_csv(f"data/Longo_2017/wrist_data_{i}.csv")
            dorsum_data.append([extract_data(data, dorsum_1_rows), extract_data(data, dorsum_2_rows)])

        return dorsum_data


    def plot_data(self, participant_num=None):
        # This plot will plot the pre positions of the dorsum, the individual judged positions, and the mean judged position
        # If a participant number is provided, it will only plot the data for that participant

        def plot_connection(ax, data_segment, pix2cm, color):
            for connection in self.connection_key:
                ax.plot([data_segment['X'][connection[0]-1]/pix2cm, data_segment['X'][connection[1]-1]/pix2cm],
                         [data_segment['Y'][connection[0]-1]/pix2cm, data_segment['Y'][connection[1]-1]/pix2cm], color)

        def plot_dorsum_data(dorsum_data, ax, test_num, participant):

            # Plot the pre positions of the dorsum
            ax.scatter(dorsum_data['pre_position']['X']/dorsum_data['pix2cm'], dorsum_data['pre_position']['Y']/dorsum_data['pix2cm'], label='Pre Position')
            plot_connection(ax, dorsum_data['pre_position'], dorsum_data['pix2cm'], 'b-')

            # Plot the individual judged positions
            # colors = ['r', 'g', 'm']
            # for i in range(1, 4):
            #     color = colors[i-1]
            #     ax.scatter(dorsum_data['indiv_judged'][str(i)]['X']/dorsum_data['pix2cm'], dorsum_data['indiv_judged'][str(i)]['Y']/dorsum_data['pix2cm'], label=f'Judged {i}', color=color)
            #     plot_connection(ax, dorsum_data['indiv_judged'][str(i)], dorsum_data['pix2cm'], color)

            # Plot the mean judged position
            ax.scatter(dorsum_data['mean_judged']['X']/dorsum_data['pix2cm'], dorsum_data['mean_judged']['Y']/dorsum_data['pix2cm'], label='Mean Judged', color='k')
            plot_connection(ax, dorsum_data['mean_judged'], dorsum_data['pix2cm'], 'k-')
            ax.legend()
            ax.set_aspect('equal', adjustable='box')
            ax.set_xlabel('X (cm)')
            ax.set_ylabel('Y (cm)')
            ax.set_title(f'Participant {participant} Test {test_num}')

        if participant_num is not None:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,6))
            dorsum_data = self.dorsum_data[participant_num-1]
            plot_dorsum_data(dorsum_data[0], ax1, 1, participant_num)
            plot_dorsum_data(dorsum_data[1], ax2, 2, participant_num)
            fig.show()

        else:
            for i in range(len(self.dorsum_data)):
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,6))
                dorsum_data = self.dorsum_data[i]
                plot_dorsum_data(dorsum_data[0], ax1, 1, i+1)
                plot_dorsum_data(dorsum_data[1], ax2, 2, i+1)
                fig.show()

        plt.show()

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
    data.plot_data()
