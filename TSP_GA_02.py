# -*- coding: utf-8 -*-
"""
Created on Sat Sep  7 14:07:33 2024

@author: Bobke
"""

import numpy as np
import matplotlib.pyplot as plt

#dest = np.array([[32,9,65,71,18,48,99,7],[54,91,34,1,87,91,63,33],["A","B","C","D","E","F","G","H"],])
cities = [["A","B","C","D","E","F","G","H"],[32,9,65,71,18,48,99,7],[54,91,34,1,87,91,63,33]]

plt.figure(1)
plt.title("TSP")
plt.plot(cities[1],cities[2], "x")
plt.grid()

for i, txt in enumerate(cities[0]):
    plt.annotate(txt,(cities[1][i],cities[2][i]))
    
plt.show()

def CityDistance(city1, city2):
    distance = ((cities[1][city1]-cities[1][city2])**2 + (cities[2][city1]-cities[2][city2])**2)**(0.5)
    return distance

def initialisation():
    chromosone = np.arange(0,7)
    np.random.shuffle(chromosone)
    return chromosone
    
def fitness(chromosone):
    fit_lvl = 0
    "Loop for distance between cities"
    for count in range(len(chromosone)):
        city1 = chromosone[count-1]
        city2 = chromosone[count]
        fit_lvl += CityDistance(city1, city2)
        #print(fit_lvl)
    "Adding in last city"
    fit_lvl += CityDistance(chromosone[0], chromosone[-1])
    #print("Distance: ", fit_lvl)
    fit_lvl = 1e3/fit_lvl
    print("Fitness: ",fit_lvl)
    return(fit_lvl)
    
def PMX(P1, P2):
    child = [""]*7
    lb = np.random.randint(0,5)
    ub = np.random.randint(lb+1,7)

        
    print(" lb: ", lb, "ub: ", ub)
    child[lb:ub] = P1[lb:ub]
    print(child)
    
    for count, gene in enumerate(child):
        if isinstance(gene, str):
            gene_loc = count
            
            double = True
            i = 0
            while double:
                gene_temp = P2[gene_loc]
                print("gene_temp: ", gene_temp)
                
                double = False
                "Check if gene is in child"
                for gene_sub in child:
                    if gene_sub == gene_temp:
                        double = True
                        print("gene found in child ")
                
                if double:
                    "Find location of gene temp in P1"
                    for count_P1, gene_P1 in enumerate(P1):
                        if gene_P1 == gene_temp:
                            gene_loc = count_P1
                            print("gene location in p1: ",gene_loc)
                else:
                    child[count] = gene_temp
                    print(child)
    return child
                    
def game_of_life(pop_size, gen_max):
    population = []
    offsprings = [0]*pop_size
    
    "initialise the game!"
    for i in range(pop_size):
        chromosone = initialisation()
        population.append(chromosone)
    #print(population)
    
    generation = 0
    
    while generation < gen_max:
        "Fitness and roulette wheel selection"
        fitnesses = []
        fit_fracs = []
        fit_frac = 0
        fit_sum = 0
        fit_max = 0
    
        
        for chromosone in population:
            fit_lvl = fitness(chromosone)
            fitnesses.append(fit_lvl)
            fit_sum += fit_lvl
            
        for fit_lvl in fitnesses:
            fit_frac += fit_lvl/fit_sum
            fit_fracs.append(fit_frac)
        print(fit_fracs)
        
        "Strongest Parent"
        for i in range(pop_size):
            if fitnesses[i] > fit_max:
                best_chromosone =  population[i]
                best_fitness =  fitnesses[i]
        print("Best Chromosone: ", best_chromosone)
        print("Best Fitness: ", best_fitness)
        
        "Making babies"
        offsprings[-1] = best_chromosone
        for i in range(pop_size-1):
            P1_choice = np.random.rand()
            P2_choice = np.random.rand()
            for j in range(pop_size):
                if  P1_choice <= fit_fracs[j]:
                    P1 = population[j]
                if  P2_choice <= fit_fracs[j]:
                    P2 = population[j]
            child = PMX(P1, P2)
            offsprings[i] = child
            #print(len(offsprings))
        generation += 1
                                        
    
chromosone_1 = initialisation()
print(chromosone_1)
fit = fitness(chromosone_1)

chromosone_2 = initialisation()
print(chromosone_2)
fit = fitness(chromosone_2)

PMX(chromosone_1, chromosone_2)

game_of_life(10, 2)

