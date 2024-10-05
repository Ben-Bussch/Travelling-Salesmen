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
    print("Distance: ", fit_lvl)
    fit_lvl = 1e3/fit_lvl
    print("Fitness: ",fit_lvl)
    return(fit_lvl)
    
def PMX(P1, P2):
    child = [""]*7
    lb = np.random.randint(0,5)
    ub = np.random.randint(lb+1,7)

        
    #print(" lb: ", lb, "ub: ", ub)
    child[lb:ub] = P1[lb:ub]
    #print(child)
    
    for count, gene in enumerate(child):
        if isinstance(gene, str):
            gene_loc = count
            
            double = True
            i = 0
            while double:
                gene_temp = P2[gene_loc]
                #print("gene_temp: ", gene_temp)
                
                double = False
                "Check if gene is in child"
                for gene_sub in child:
                    if gene_sub == gene_temp:
                        double = True
                        #print("gene found in child ")
                
                if double:
                    "Find location of gene temp in P1"
                    for count_P1, gene_P1 in enumerate(P1):
                        if gene_P1 == gene_temp:
                            gene_loc = count_P1
                            #print("gene location in p1: ",gene_loc)
                else:
                    child[count] = gene_temp
                    #print(child)
    return child
                    
def game_of_life(pop_size, gen_max):
    population = []
    offsprings = [0]*pop_size
    
    "initialise the game!"
    for i in range(pop_size):
        chromosone = initialisation()
        population.append(chromosone)
    print(population)
    
    g_fit_best = []
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
        print("Fit frac :", fit_fracs)
        
        "Strongest Parent"
        for i in range(pop_size):
            if fitnesses[i] > fit_max:
                fit_max = fitnesses[i]
                best_chromosone =  population[i]
                best_fitness =  fitnesses[i]
        print("Best Chromosone: ", best_chromosone)
        print("Best Fitness: ", best_fitness)
        g_fit_best.append(1e3/best_fitness)
        
        "Making babies"
        offsprings[-1] = best_chromosone
        for i in range(pop_size-1):
            P1_choice = np.random.rand()
            P2_choice = np.random.rand()
            
            if  P1_choice <= fit_fracs[0]:
                    P1 = population[0]
                    #print(P1_choice, fit_fracs[j])
            if  P2_choice <= fit_fracs[0]:
                P2 = population[0]
                
            for j in range(pop_size-1):
                if  P1_choice >= fit_fracs[j]:
                    P1 = population[j+1]
                    #print(P1_choice, fit_fracs[j])
                    
                if  P2_choice >= fit_fracs[j]:
                    P2 = population[j+1]
                    
            
            child = PMX(P1, P2)
            offsprings[i] = child
            #print(len(offsprings))
            
        population = offsprings        
        generation += 1
    
    
    gens = np.arange(0, gen_max)
    plt.figure(2)
    plt.title("Fitness vs iteration")
    plt.plot(gens, g_fit_best)
    plt.grid()
    plt.xlabel("Iteration")
    plt.ylabel("Path Distance")
                                        
    

game_of_life(50, 50)

