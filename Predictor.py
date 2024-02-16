import pygame, sys, random
import matplotlib.pyplot as plt
import os
import time
import neat
from drawnow import drawnow
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
        xList = [x for x, _, _  in plot_data if x is not None]
        yaList = [y for _, y, _ in plot_data if y is not None]
        ypList = [y for _, _, y in plot_data if y is not None]
        if len(xList) != len(yaList) or len(xList) != len(ypList):
            raise ValueError("xList, yaList, and ypList must have the same length.")

        # Plot `yaList` with a different marker and label
        plt.plot(xList, yaList, marker='^', label='yaList')

        # Plot `ypList` with another marker and label
        plt.plot(xList, ypList, marker='s', label='ypList')

        x, y = xList[-1], ypList[-1]
        plt.text(x + 0.1, y, f"Generation: {gen}", ha='center', fontsize=8)
        # Set plot labels and title
        plt.xlabel("Stocks")
        plt.ylabel("data point")
        plt.title("Tesla data plot")

        
plot_data=[]
gen=0
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
    past_pred=""
    past_act=""
    plt.ion() # enable interactivity
    plt.figure() # make a figure
    past_10_close=[]
    for index, row in dataset.iterrows():
        # Get the past five values of rows["Close"] into a list
        past_10_close.append(row["Close"])
        if len(past_10_close) > 10:
            past_10_close = past_10_close[-10:]
            for genome, net in zip(ge, nets):
                    next_index = index + 1
                    next_close = dataset.iloc[next_index]["Close"]
                    output = net.activate(tuple(past_10_close))
                    prediction = output[0]
                    if next_index < len(dataset):
                        
                        if abs(prediction - next_close) <= 0.5:
                            # fitness_increment = max(0, 1 - (abs(prediction - next_close) / 3))
                            # genome.fitness += fitness_increment
                            genome.fitness += 1
                        else:
                            genome.fitness -= 1
                    if genome.fitness<=0:
                        ge.pop(ge.index(genome)) 
                    plot_data.append([(index + 1), row["Close"], prediction])
                    drawnow(makeFig)
                    plt.pause(0.001)
                    # Clear data for the next iteration
                    print("Generation: ", gen,"Predicted: ", prediction, "Actual: ", row["Close"])
            # sorted_genomes = sorted(ge, key=lambda genome: genome.fitness, reverse=True)
            # top_3_genomes = sorted_genomes[:3]  # Select the top 3 genomes
            # ge = top_3_genomes
    plt.close("all")
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