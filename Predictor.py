import pygame, sys, random
import matplotlib.pyplot as plt
import os
import time
import neat
from drawnow import drawnow
gen=0
#load csv dataset for training from /archive/TSLA.csv use parsing if necessary
import pandas as pd
import numpy as np


def load_dataset(csv_file_path):
    """
    Load and parse the CSV dataset for training.
    
    :param csv_file_path: The file path to the CSV file.
    :return: A pandas DataFrame containing the dataset.
    """
    try:
        # Load the dataset
        data = pd.read_csv(csv_file_path)
        print(f"Dataset loaded successfully with {len(data)} entries.")
        return data
    except Exception as e:
        print(f"An error occurred while loading the dataset: {e}")
        return None

# Usage
dataset = load_dataset('./archive/TSLA.csv')
def makeFig():
        print(plot_data)
        xList = [x for x, _ in plot_data if x is not None and _ is not None]
        yList = [y for _, y in plot_data if y is not None and _ is not None]

        plt.plot(xList, yList, marker='o', label='Data') # I think you meant this
plot_data=[]
def main(genomes,config):
    global gen
    ge = []
    nets = []
    global gen
    gen+=1
    for genome_id, genome in genomes:
        genome.fitness = 0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        nets.append(net)
        ge.append(genome)
    # Create a figure and axes for all genomes
    global plot_data 
    plot_data = []
    plt.ion() # enable interactivity
    plt.figure() # make a figure
    for index, row in dataset.iterrows():
        for genome, net in zip(ge, nets):
            output = net.activate((row["Open"], row["Close"], row["High"], row["Low"]))
            decision = output.index(max(output))
            plot_data.append([(index + 1), row["Close"]])
            if decision == 0:
                prediction="Down"
            elif decision == 1:
                prediction="Up"
            else:
                genome.fitness -= 1
            next_open = dataset["Open"].shift(-1).iloc[index] if index < len(dataset) - 1 else None
            next_close = dataset["Close"].shift(-1).iloc[index] if index < len(dataset) - 1 else None

            print(next_open, next_close) 
            if next_close > next_open:
                actual_next="Up"
            else:
                actual_next="Down"
            if prediction == actual_next:
                genome.fitness += 1
            else:
                genome.fitness -= 1
            drawnow(makeFig)
            plt.pause(0.001)
            # Clear data for the next iteration
            print("Generation: ", gen,"Predicted: ", prediction, "Actual: ", actual_next)
def run(config_file):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    #p.add_reporter(neat.Checkpointer(5))

    # Run for up to 50 generations.
    global iter
    iter=50
    winner = p.run(main, iter)

    # show final stats
    print('\nBest genome:\n{!s}'.format(winner))
if __name__ == '__main__':
    
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'neat-config.txt')
    run(config_path)