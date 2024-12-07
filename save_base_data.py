from data_handler import Data2017
import matplotlib.pyplot as plt

if __name__ == "__main__":
    data = Data2017()
    
    for participant in range(1, 13):
        fig = data.plot_data(participant)
        fig.savefig(f'figures/participant_{participant}.png', transparent=False)
