import data_handler

if __name__ == "__main__":
    data = data_handler.import_2016()
    
    # Make a plot of the data
    data_handler.plot_data(data)
