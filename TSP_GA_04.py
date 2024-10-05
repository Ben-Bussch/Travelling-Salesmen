# -*- coding: utf-8 -*-
"""
Created on Sat Sep  7 14:07:33 2024

@author: Bobke
"""

import numpy as np
import matplotlib.pyplot as plt

#dest = np.array([[32,9,65,71,18,48,99,7],[54,91,34,1,87,91,63,33],["A","B","C","D","E","F","G","H"],])
cities = [["A","B","C","D","E","F","G","H","Ellen","Josh","Freddie","Robbie", "Jess","Sitara", "Aaron", "Alex"],[32,9,65,71,18,48,99,7,50,32,68,95,43,83,98,4],[54,91,34,1,87,91,63,33,50,9,62,81,22,8,19,57]]
city_length = len(cities[0])
plt.figure(1)
plt.title("TSP")
plt.plot(cities[1],cities[2], "x")
plt.grid()

print("Number of cities: ", city_length)

for i, txt in enumerate(cities[0]):
    plt.annotate(txt,(cities[1][i],cities[2][i]))
    
plt.show()

def CityDistance(city1, city2):
    distance = ((cities[1][city1]-cities[1][city2])**2 + (cities[2][city1]-cities[2][city2])**2)**(0.5)
    return distance

def initialisation():
    chromosone = np.arange(0,city_length)
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
    #print("Fitness: ",fit_lvl)
    return(fit_lvl)
    
def PMX(P1, P2):
    child = [""]*city_length
    lb = np.random.randint(0,city_length-2)
    ub = np.random.randint(lb+1,city_length)

        
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

def mutation(offspring):
    mutation_rate = 0.05
    mutation_lottery = np.random.rand()
    if mutation_lottery <= mutation_rate:
        #print("Mutate offspring before:", offspring)
        mutation_1 = np.random.randint(0,city_length)
        mutation_2 = np.random.randint(0,city_length)
        gene_temp = offspring[mutation_1]
        offspring[mutation_1] = offspring[mutation_2]
        offspring[mutation_2] = gene_temp
        #print("Mutate offspring after:", offspring)
        
    return offspring
    
                    
def game_of_life(pop_size, gen_max):
    population = []
    offsprings = [0]*pop_size
    
    "initialise the game!"
    for i in range(pop_size):
        chromosone = initialisation()
        population.append(chromosone)
    #print(population)
    
    g_fit_best = []
    g_fit_avg = []
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
        
        g_fit_avg.append(1e3/(fit_sum/pop_size))    
        
        for fit_lvl in fitnesses:
            fit_frac += fit_lvl/fit_sum
            fit_fracs.append(fit_frac)
        #print("Fit frac :", fit_fracs)
        
        "Strongest Parent"
        for i in range(pop_size):
            if fitnesses[i] > fit_max:
                fit_max = fitnesses[i]
                best_chromosone =  population[i]
                best_fitness =  fitnesses[i]
        #print("Best Chromosone: ", best_chromosone)
        #print("Best Fitness: ", best_fitness)
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
            mutated_child = mutation(child)
            offsprings[i] = mutated_child
            
            #print(len(offsprings))
            
        population = offsprings        
        generation += 1
        print("Generation: ", generation)
        print("Best Chromosone: ", best_chromosone)
        print("Best Fitness: ", best_fitness)
    
    #print(population)
    gens = np.arange(0, gen_max)
    plt.figure(2)
    plt.title("Fitness vs iteration")
    plt.plot(gens, g_fit_best, label= "best")
    plt.plot(gens, g_fit_avg, label= "Avg")
    plt.grid()
    plt.xlabel("Iteration")
    plt.ylabel("Path Distance")
    plt.legend()
    
    print("Best Distance: ", 1e3/best_fitness)
    print("Best City Order:")
    for order in best_chromosone:
        print(cities[0][order])
                                        
    

game_of_life(50, 200)

