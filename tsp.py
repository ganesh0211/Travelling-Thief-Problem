# -*- coding: utf-8 -*-

import random
import math
import numpy
import matplotlib.pyplot as plt
from collections import deque


def decide_to(p):
    '''
    Randomly returns True/False based on probability.
    
    Args:
        p   :float of [0,1]; probability of the return value being True
    Returns:
        True with the probability of p and False with the probability of (1-p)
    '''
    r = random.uniform(0, 1)
    return r <= p


def mutate(ind, N):
    '''
    Mutates the individual ind by swapping two genomes.
    Args:
        ind : |N| list/numpy.array of chromosome/individual
        N   : length of ind 
    Returns:
        None
    '''
    i = random.randrange(0, N)
    j = random.randrange(0, N)
    ind[i], ind[j] = ind[j], ind[i]
    return


def crossover(father, mother, child1, child2, N):
    '''
    Performs the Order 1 Crossover and produces two children by modifying child1 and child2.
    
    Args:
        father: |N| list/numpy.array representing the father chromosome
        mother: |N| list/numpy.array representing the mother chromosome
        child1: |N| list/numpy.array representing the first child
        child2: |N| list/numpy.array representing the second child
        N: length of all the input chromosomes
    Returns:
        None
    '''
    # randomly choosing the crossover range
    i = random.randrange(0, N)
    j = random.randrange(0, N)
    if i > j:
        i, j = j, i

    # keeping track of the exact intervals
    check1 = numpy.zeros(N)
    check2 = numpy.zeros(N)

    for x in range(i, j + 1):
        child1[x] = father[x]  # Redundant because child1 is initialized as the father
        child2[x] = mother[x]  # Redundant because child2 is initialized as the mother
        check1[father[x]] = 1
        check2[mother[x]] = 1

    # copying the remaining genomes sequentially for child1
    x = 0
    index = 0 + ((i == 0) * (j + 1))
    while x < N and index < N:
        if not check1[mother[x]]:
            child1[index] = mother[x]
            index += 1
        if index == i:
            index = j + 1
        x += 1
    # copying the remaining genomes sequentially for child2
    x = 0
    index = 0 + ((i == 0) * (j + 1))
    while x < N and index < N:
        if not check2[father[x]]:
            child2[index] = father[x]
            index += 1
        if index == i:
            index = j + 1
        x += 1
    return


def distance(points, order, N):
    '''
    Computes the euclidean distance of a cycle
    
    Args:
        points  : |Nx2| list/numpy.array of coordinates of points
        order   : |N|   list/numpy.array of ordering of points (zero indexed)
    Returns:
        euclidean distance of points from point0 to point1 to ... pointN back to point0
    Examples:
        >>> distance([[0,0],[0,4],[3,4]],[0,1,2],3)
        12.0
    '''
    x0 = points[0][0]
    y0 = points[0][1]
    xi = points[order[0]][0]
    yi = points[order[0]][1]
    s = math.sqrt((x0 - xi) ** 2 + (y0 - yi) ** 2)
    for i in range(1, N):
        x1 = points[order[i - 1]][0]
        y1 = points[order[i - 1]][1]
        x2 = points[order[i]][0]
        y2 = points[order[i]][1]
        s += round(math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2))

    return s


def init_population(K, N):
    '''
    Initializes the population of K chromosomes with N genome for each chromosome by modifying pop.
    
    Args:
        K: number of chromosomes
        N: number of genomes in each chromosome
    Returns:
        |KxN| numpy.array of K chromosomes with N genome for each
    '''
    pop = numpy.zeros((K, N), dtype=numpy.int32)
    # each chromosome is a shuffle of sequence 1:N
    seq = list(range(N))
    for i in range(K):
        random.shuffle(seq)
        pop[i] = seq
    return pop


def compute_population_fitness(pop, points, K, N):
    '''
    Computes the fitness for each chromosome in the population by modifying fit.
    (Fitness of each chromosome is the negative of its cycle distance.)
    
    Args:
        pop: |KxN| list/numpy.array of K chromosomes with N genome for each
        points: |Nx2| list/numpy.array of coordinates of points
        K: number of chromosomes
        N: number of genomes in each chromosome
    Returns:
        fit, a |K| list/numpy.array of floats where fit[k] = fitness of pop[k].
    Examples:
        >>> compute_population_fitness([[0,3,1,2],[2,1,0,3]],[[0,0],[0,4],[3,4],[3,0]],2,4)
        array([-16., -14.])
    '''
    fit = numpy.zeros(K)
    for k in range(K):
        # fitness of each chromosome is the negative of its cycle distance
        fit[k] = -distance(points, pop[k], N)
    return fit


def find_cumulative_distribution(arr, K):
    '''
    Computes cumulative distribution (percentages) of arr.
    
    Args:
        arr: |K| numpy.array of numbers.
        K: length of arr
    Returns:
        cd, |K| numpy.array of floats containing the cumulative distributions
        where cd[i] is the probability that a uniform random number in [0,arr.sum()] is
        less than arr[:i].sum()
    Examples:
        >>> find_cumulative_distribution(numpy.array([4,2,2]),3)
        array([ 0.5 ,  0.75,  1.  ])
    '''
    cd = numpy.zeros(K)
    acc = 0
    s = arr.sum()
    for i in range(K):
        acc += arr[i] / s
        cd[i] = acc
    return cd


def select_parent(fitness, K):
    '''
    Select and index for parent based on fitness of each chromosome using the roulette wheel technique.
    
    Args:
        fitness: |K| list/numpy.array of numbers representing the fitness for each chromosome
        K: length of fitness
    Returns:
        index of the randomly selected parent
    '''
    local_absolute_fitness = fitness - fitness.min()  # now worst individual has fitness 0
    # implementation of roulette wheel technique for choosing a random number representing
    # the parent by using the cumulative probability of each element of the fitness list.
    cd = find_cumulative_distribution(local_absolute_fitness, K)
    roulette = random.uniform(0, 1)
    ind = 0
    while roulette > cd[ind]:
        ind += 1
    return ind


def create_new_population(pop, fitness, K, N, crossover_probability, mutation_probability):
    '''
    Creates a new population of K chromosomes of N genomes
    by crossovers and mutations over the current population.
    
    Args:
        pop: |KxN| list/numpy.array of K chromosomes with N genome for each chromosome
         representing the current population
        fitness: |K| list/numpy.array of fitness of each chromosome in pop
        K: number of chromosomes
        N: number of genomes in each chromosome
        crossover_probability: float in [0,1] representing crossover probability
        mutation_probability: float in [0,1] representing mutation probability
    Returns:
        |KxN| list/numpy.array of K chromosomes with N genome for each chromosome
        representing the new population
    '''
    new_pop = numpy.zeros((K, N), dtype=numpy.int32)
    for k in range(K // 2):  # 2 children are created in each iteration
        father_ind = select_parent(fitness, K)
        mother_ind = select_parent(fitness, K)

        father = pop[father_ind]
        mother = pop[mother_ind]
        child1 = father.copy()
        child2 = mother.copy()

        if decide_to(crossover_probability):
            crossover(father, mother, child1, child2, N)
        if decide_to(mutation_probability):
            mutate(child1, N)
        if decide_to(mutation_probability):
            mutate(child2, N)

        new_pop[k * 2] = child1
        new_pop[k * 2 + 1] = child2
    return new_pop


def find_best_individual(pop, fitness, best_individual, best_fit):
    '''
    Finds the best individual and the fitness of that individual in all the generations.
    
    Args:
        pop: |KxN| list/numpy.array of K chromosomes with N genome for each
        fitness: |K| list/numpy.array of numbers representing the fitness for each chromosome
        best_individual: |N| list/numpy.array representing the best individual/chromosome
        so far excluding the current population.
        best_fit: number representing the fitness of best_individual
    Returns:
            {best individual so far},{fitness of best individual so far}
    '''
    current_best_index = fitness.argmax()
    current_best_fit = fitness[current_best_index]
    current_best_individual = pop[current_best_index]

    if best_fit < current_best_fit:
        return current_best_individual, current_best_fit
    else:
        return best_individual, best_fit


def read_input(path, N):
    '''
    Reads the first N lines of the .txt file denoted by path
    containing the coordinates of the points in the following format:
    x_1 y_1
    x_2 y_2
    ...
    
    Args:
        path: string that indicates the path to the text file containing coordinates of n points
    Returns:
        |Nx2| numpy.array of coordinates of points
    '''
    points = numpy.zeros((N, 2))
    file = open(path)
    lines = file.readlines()
    lines = [x.replace(',',' ') for x in lines]
    file.close()
    for i in range(N):
        points[i][0], points[i][1] = map(int, lines[i].split())
    return points


def plot_individual_path(individual, points, title, index):
    '''
    Plots individual cycle in the index of a 3x5 plot
     
    Args:
        individual:  |N| list/numpy.array of a chromosome
        points: |Nx2| list/numpy.array of coordinates of points
        title: title of the plot
        index: integer in [1,15] denoting the position of the plot in a 3x5 matplotlib subplots.
    Returns:
        None
    '''
    x = []
    y = []
    for i in individual:
        x.append(points[i][0])
        y.append(points[i][1])
    x.append(x[0])
    y.append(y[0])

    plt.subplot(3, 5, index)
    plt.title(title)
    plt.xticks([], [])
    plt.yticks([], [])
    plt.plot(x, y, 'r*')
    plt.plot(x, y, 'g--')
    return


def plot_results(best_last_15, points):
    '''
    Plots and displays the best last 15 chromosomes generated through the generations.
    
    Args:
        best_last_15: |M| deque/list of (A,B) where A is one of the best chromosomes and B is its cycle distance
          and M<=15
        points: |Nx2| list/numpy.array of coordinates of points
    Returns:
        None
    '''
    for i in range(0,len(best_last_15)):
        plot_individual_path(best_last_15[i][0], points, str(round(best_last_15[i][1], 2)), i+1)
    plt.show()
    return


def TSP_genetic(n, k, max_generation, crossover_probability, mutation_probability, path):
    '''
    Solves the Traveling Sales Person Problem using genetic algorithm with chromosomes decoded
    as cycles (solutions) of traveling Order 1 crossover, Swap mutation, complete generation 
    replacement, Roulette Wheel Technique for choosing parents and negative of cycle distance for fitness.
    
    
    Args:
        n: integer denoting the number of points and also the number of genome of each chromosom 
        k: integer denoting the number of chromosomes in each population
        max_generation: integer denoting the maximum generation (iterations) of the algorithm
        crossover_probability: float in [0,1] denoting the crossover probability
        mutation_probability: float in [0,1] denoting the mutation probability
        path: string that indicates the path to the text file containing coordinates of n points
    Returns:
        None
    '''
    points = read_input(path, n)
    population = init_population(k, n)
    best_individual = population[0]  # arbitrary choose a chromosome for initialization of best individual.
    best_fitness = -distance(points, best_individual, n)  # setting -distance as fitness for best individual.
    old_best_fitness = best_fitness
    best_last_15 = deque([], maxlen=15)  # queue with fixed size of 15

    for generation in range(1, max_generation + 1):
        # 1. We compute the fitness of each individual in current population.
        fitness = compute_population_fitness(population, points, k, n)
        # 2. We obtain the best individual so far together with its fitness.
        best_individual, best_fitness = find_best_individual(population, fitness, best_individual, best_fitness)
        # 3. We save the best last 15 individuals for plotting
        if old_best_fitness != best_fitness:
            old_best_fitness = best_fitness
            best_last_15.append((best_individual.copy(), -best_fitness))
        # 4. We create the next generation
        population = create_new_population(population, fitness, k, n, crossover_probability, mutation_probability)
        # 5. Prints best distance so far
        print(-best_fitness, '\t', generation)

    solution = best_individual
    cycle_distance = -best_fitness


    print(cycle_distance)
    #print(solution+1)
    #plot_results(best_last_15, points)
    return solution+1


# General Parameters
#n = 279
#k = 120
#max_generation = 500
#crossover_probability = 0.99
#mutation_probability = 0.01

#path = 'nodes.txt'
#TSP_genetic(n, k, max_generation, crossover_probability, mutation_probability, path)
