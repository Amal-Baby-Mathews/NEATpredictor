import pygame, sys, random
import matplotlib.pyplot as plt
import os
import time
import neat
from drawnow import drawnow
#load csv dataset for training from /archive/TSLA.csv use parsing if necessary
import pandas as pd
import numpy as np
from collections import defaultdict

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
        xList = [x for x, _ in plot_data if x is not None]
        yaList = [y for _, y in plot_data if y is not None]
        id = glob_id  # Assuming this is defined somewhere
            
        # Check if id exists in id_to_predictions:
        if id in id_to_predictions:
            predlist = id_to_predictions[id]

            # Pad predlist with zeros to match yaList's length:
            padding_length = len(yaList) - len(predlist)

            # Ensure the actual prediction is at the end (if padding_length > 0):
            if padding_length > 0:
                predlist = predlist + [0] * padding_length
                predlist[-1] = id_to_predictions[id][-1]  # Assuming the last element is the prediction

            # Print and use the lists:
            print(xList, yaList, predlist)
        else:
            print("ID not found in predictions dictionary.")
        print(xList, yaList, predlist,id)
        if len(xList) != len(yaList) or len(xList) != len(predlist):
            raise ValueError("xList, yaList, and ypList must have the same length.")
        # Plot `yaList` with a different marker and label
        plt.plot(xList, yaList, marker='^', label='yaList')
        # Plot `ypList` with another marker and label
        # plt.plot(xList, ypList, marker='s', label='ypList')
        plt.plot(xList, predlist, marker='s', label=str(id))
        x, y = xList[-1], predlist[-1]
        plt.text(x + 0.1, y, f"Generation: {gen}", ha='center', fontsize=8)
        # Set plot labels and title
        plt.xlabel("Stocks")
        plt.ylabel("data point")
        plt.title("Tesla data plot")
plot_data=[]
id_to_predictions = defaultdict(list)
gen=0
def main(genomes,config):
    global glob_id
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
    global id_to_predictions
    id_to_predictions = defaultdict(list)
    plt.ion() # enable interactivity
    plt.figure() # make a figure
    past_10_close=[]
    for index, row in dataset.iterrows():

        # Get the past five values of rows["Close"] into a list
        past_10_close.append(row["Close"])
        if len(past_10_close) > 5:
            plot_data.append([(index + 1), row["Close"]])
            past_10_close = past_10_close[-5:]
            for genome, net in zip(ge, nets):
                    next_index = index + 1
                    next_close = dataset.iloc[next_index]["Close"]
                    output = net.activate(tuple(past_10_close))
                    prediction = output[0]
                    glob_id=genome.key
                    id_to_predictions[genome.key].append(prediction)
                    if next_index < len(dataset):
                        
                        if (prediction - next_close) <= 5 and (prediction - next_close) >=0:
                            fitness_increment = max(0, 2 - (abs(prediction - next_close) / 3))
                            genome.fitness += fitness_increment
                            # genome.fitness += 1.9
                        else:
                            genome.fitness -= 0.5
                    if genome.fitness<=0:
                        ge.pop(ge.index(genome))
                    drawnow(makeFig)
                    plt.pause(0.001)
                    # Clear data for the next iteration
                    print("Generation: ", gen,"Id: ", genome.key,"Predicted: ", prediction, "Actual: ", row["Close"] )
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