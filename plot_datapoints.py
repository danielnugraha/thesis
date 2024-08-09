import csv
import matplotlib.pyplot as plt

# Read the CSV file
csv_data = []
with open('_static/mvs_0.1_indices.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        csv_data.append([float(item.strip('[]')) for item in row])

# Experiment configuration
experiments_config = {
    3: 6,   # First experiment: 6 rows per round
    5: 10,  # Second experiment: 10 rows per round
    10: 20  # Third experiment: 20 rows per round
}

# Number of rounds
num_rounds = 50

# Initialize the dictionary to hold all experiments data
experiments_data = {}

# Index to keep track of the starting point in csv_data
start_index = 0

# Process each experiment based on its configuration
for key, rows_per_round in experiments_config.items():
    experiment_data = {}
    
    for round_num in range(num_rounds):
        end_index = start_index + rows_per_round
        
        round_data = csv_data[start_index:end_index]
        experiment_data[round_num + 1] = {'x': [], 'y': []}
        
        for i, row in enumerate(round_data):
            if i % 2 == 0:
                experiment_data[round_num + 1]['x'].append(row)
            else:
                experiment_data[round_num + 1]['y'].append(row)
        
        start_index = end_index
    
    experiments_data[key] = experiment_data

# Rounds to plot
rounds_to_plot = [1, 5, 10, 30, 50]

# Plotting function
def plot_round_data(round_data, experiment_key, round_number):
    x_data = [x for sublist in round_data['x'] for x in sublist]
    y_data = [y for sublist in round_data['y'] for y in sublist]

    plt.figure()
    plt.hexbin(x_data, y_data, gridsize=30, cmap='Blues')
    plt.colorbar(label='Count')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(f'Federated MVS wine quality {experiment_key} clients, Round {round_number}')
    plt.grid(True)
    plt.savefig(f'_static/experiment_{experiment_key}_round_{round_number}.png')
    plt.close()

# Plot the specified rounds for each experiment
for experiment_key, rounds in experiments_data.items():
    for round_number in rounds_to_plot:
        plot_round_data(rounds[round_number], experiment_key, round_number)
