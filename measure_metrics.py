import os
import re
from collections import defaultdict
from utils import metrics_args_parser
import matplotlib.pyplot as plt
from typing import Dict, List

def plot_result(data: Dict[float, List[float]], title: str, filename: str):
    plt.figure(figsize=(15, 10))
    for key, values in data.items():
        plt.plot(values, linewidth=2, label=key)

    fontsize = 18
    plt.legend(fontsize=fontsize, loc='lower left', bbox_to_anchor=(1, 0))
    plt.grid()
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.title(title, fontsize=20)
    plt.xlabel('Rounds', fontsize=fontsize)
    plt.ylabel('Evaluations', fontsize=fontsize)

    plt.tight_layout()
    plt.savefig(filename)

    

def read_convert_and_delete_files(dataset_prefix: str) -> None:
    directory = "./_static"
    output_file = f"{directory}/{dataset_prefix}.txt"

    results = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    
    pattern = re.compile(rf'{re.escape(dataset_prefix)}_(\w+)_(\w+)_(0\.\d|1\.0)_(\d+)\.txt')
    
    for filename in os.listdir(directory):
        match = pattern.match(filename)
        if match:
            sampling_method = match.group(1)
            partitioner_type = match.group(2)
            sampling_fraction = float(match.group(3))
            number_of_clients = int(match.group(4))
            
            with open(os.path.join(directory, filename), 'r') as file:
                content = file.read().strip()
                values = content.split(',')
            
            for value in values:
                results[number_of_clients][sampling_method][partitioner_type][sampling_fraction].append(value)
            
            os.remove(os.path.join(directory, filename))
    
    with open(output_file, 'w') as file:
        print(results)
        for number_of_clients, methods in results.items():
            for sampling_method, partitioners in methods.items():
                for partitioner_type, fractions in partitioners.items():
                    first_values_list = next(iter(fractions.values()), [])
                    size_of_content = len(first_values_list)

                    if number_of_clients == 1:
                        title = f'Centralized {sampling_method.upper()} - {dataset_prefix.capitalize()} - {size_of_content} rounds'
                        filename = f'{directory}/{dataset_prefix}_{sampling_method}_centralized_{size_of_content}.png'
                    else:
                        title = f'Federated {sampling_method.upper()} - {dataset_prefix.capitalize()} - {number_of_clients} clients {partitioner_type} local evaluation - {size_of_content} rounds'
                        filename = f'{directory}/{dataset_prefix}_{sampling_method}_{number_of_clients}_{partitioner_type}_{size_of_content}.png'

                    file.write(f'{title}\n')
                    file.write('{\n')

                    plot_result(fractions, title, filename)

                    for sampling_fraction, contents in sorted(fractions.items()):
                        file.write(f'    {sampling_fraction}: {','.join(map(str, contents))},\n')
                            
                    file.write('}\n\n')

args = metrics_args_parser()

read_convert_and_delete_files(args.dataset)
