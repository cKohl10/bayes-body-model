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

class Data2016(Data):
    def __init__(self):
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

        # Convert Excel column letters to numeric indices
        def excel_col_to_num(col_letter):
            """Convert Excel column letter to corresponding numeric index (0-based)"""
            num = 0
            for c in col_letter:
                num = num * 26 + (ord(c.upper()) - ord('A')) + 1
            return num - 1

        # Column definitions
        mean_col = excel_col_to_num("D")  # 3
        std_col = excel_col_to_num("E")   # 4
        sample_start = excel_col_to_num("I")    # 8
        sample_end = excel_col_to_num("AK")     # 36
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
        # Plot the data points for each distance and direction
        for key in self.keys:
            for direction in self.directions:
                for inference in self.inference:
                    
        plt.legend()
        plt.show()



if __name__ == "__main__":
    data = Data2016()
    data.plot_data()
