# -*- coding: utf-8 -*-
"""
Created on Sat Dec 19 18:29:52 2020

@author: Aarsh
"""

#from typing import List,Callable, Tuple, Optional
#from collections import namedtuple
from random import random,choices,randint,randrange
import string
from typing import List, Callable
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tsp import TSP_genetic

#--------------------------------INPUTS---------------------------------------

# Knapsack
parent_size = 2     # no need to change
mutation_prob = 0.5
generatoin_limit = 400
population_size = 15
fitness_limit = 0   # not used anywhere
top_sol = 2

# TSP
max_generation = 10
crossover_probability = 0.99
mutation_probability = 0.01

# ---------------------Txt file reading start here----------------------------

# open data.txt file
with open("data_1.txt","r") as f: 
    lines = f.readlines()
    lines = lines[2:]
    lines = [x.strip() for x in lines]
    lines = [x.replace(':','') for x in lines]  # removes all semi colon
    lines = [x.replace('(','') for x in lines]  # removes all (
    lines = [x.replace(')','') for x in lines]  # removes all )
    lines = [x.split('\t') for x in lines]      # splits with tabs
    
    
    for i in range(len(lines)):
        
        # starting of the reading of coords
        if lines[i][0] == "NODE_COORD_SECTION": 
            start=i
        
        # Number of data points
        if lines[i][0] == "DIMENSION":
            dimension=int(lines[i][1])
            
        # Number of total items
        if lines[i][0] == "NUMBER OF ITEMS ":
            items=int(lines[i][1])
            
        # Knapsack capacity
        if lines[i][0] == "CAPACITY OF KNAPSACK ":
            weight_limit=int(lines[i][1])
            
        # Minimum velocity
        if lines[i][0] == "MIN SPEED ":
            vmin = float(lines[i][1])   
            
        # Maximum velocity
        if lines[i][0] == "MAX SPEED ":
            vmax = float(lines[i][1])
            
        # Rent ratio
        if lines[i][0] == "RENTING RATIO ":
            rent_ratio = float(lines[i][1])
        
        
    # Assumed all cities has same number of items
    item_per_city = int(items/(dimension-1))
    # reading locations from data
    loc = np.array(lines[start+1:start+dimension+1])
    loc[loc == ''] = 0
    loc = loc.astype(np.int)
    # sorting location by index points
    loc = loc[np.argsort(loc[:,0])]
    # starting point (x,y)
    start_loc = [loc[0,1],loc[0,2]]
    # all X coords
    x_loc = loc[:,1]
    # all Y coords
    y_loc = loc[:,2]
    xy_loc = np.transpose(np.matrix([x_loc,y_loc]))
    
    # writting XY location in another text for TSP
    with open('nodes.txt','wb') as f:
        for line in xy_loc:
            np.savetxt(f,xy_loc,delimiter=',',fmt='%i')
            
    # Distance matrix from point to point        
    dist_mat = np.zeros((dimension,dimension))
    #dist_cont = np.zeros(items)
    
    
    for i in range(0,len(loc)):
        for j in range(0,len(loc)):
            dist_mat[i][j] = round(np.sqrt((loc[i][1]-loc[j][1])**2 + (loc[i][2]-loc[j][2])**2))
            #dist_cont[i*item_per_city:(i+1)*item_per_city] = np.square((loc[0][1]-loc[i][1])) + np.square((loc[0][2]-loc[i][2]))
    # dataframe of locations
    df_loc = pd.DataFrame(loc,columns=lines[start][1].split(','))
    #loc_sort = df_loc.sort_values(by=['INDEX'])
    
    # reading item list
    bag = np.array(lines[start+dimension+2 : start+dimension+items+2])
    bag[bag == ''] = 0
    bag = bag.astype(np.int)
    
    # sorting according to allocated cities (last column)
    bag = bag[np.argsort(bag[:,-1])]
    
    # Profit array for each items after sorting
    # profit[0:item_per_city] is profit for items from 2nd city
    profit = bag[:,1]
    
    # weights array for each items after sorting
    # weights[0:item_per_city] is weights for items from 2nd city
    weights = bag[:,2]
    
    # node array for each items after sorting
    # node[0:item_per_city] is node-2 == [2,2,2,2,2]
    node = bag[:,3]
    
    # combining location and item data
    loc_bag = np.insert(bag,[np.ma.size(bag,1)],[0,0],axis=1)
    
    # creating dataframe of combined data
    df_loc_bag = pd.DataFrame(loc_bag,columns=[lines[start+dimension+1][1].split(',')+lines[start][1].split(',')[1:]])
    #bag_sort = df_bag.sort_values(by=[' ASSIGNED NODE NUMBER'])
    
    for i in range(len(df_loc_bag)):
        a = df_loc_bag.loc[i].at[' ASSIGNED NODE NUMBER'].item()
        df_loc_bag.loc[i].at[' X'] = loc[np.where(loc[:,0]==a)[0][0],1]
        df_loc_bag.loc[i].at[' Y'] = loc[np.where(loc[:,0]==a)[0][0],2]
        

# ------------------------Text reading end here-------------------------------



#---------------------------- TSP GA ----------------------------------

n = dimension-1
k = 120
path = 'nodes.txt'
best_path = TSP_genetic(n, k, max_generation, crossover_probability, mutation_probability, path)
# Best route according to TSP solver starting from position 0
best_path = np.insert(best_path,0,0)




#----Creating profit and weights according to their index from best path------

# Node weights is weight is just extended array of with same distance
# raising from smalles dist to largest according to best_path
node_weights = np.zeros((len(best_path)-1)*item_per_city)
dist=0

# new profit are re-aranged profit based on best path
new_profit = np.zeros(len(profit))

# new weights are re-aranged weights based on best path
new_weights = np.zeros(len(weights))

# filling above array from best path
for i in range(len(best_path)-1):

    x1 = x_loc[best_path[i]]
    x2 = x_loc[best_path[i+1]]
    y1 = y_loc[best_path[i]]
    y2 = y_loc[best_path[i+1]]
    dist += round(np.sqrt((x1-x2)**2 + (y1-y2)**2))
    start = i*item_per_city
    end = start + item_per_city
    node_weights[start : end ] = dist
    new_profit[start : end] = profit[(best_path[i+1]*item_per_city)-5 : (best_path[i+1]*item_per_city)]
    new_weights[start : end] = weights[(best_path[i+1]*item_per_city)-5 : (best_path[i+1]*item_per_city)]



# ---------------------------------Knapsack-----------------------------------

Inputs = [new_weights, new_profit, node_weights, weight_limit, 
          vmax, vmin, best_path, item_per_city, parent_size,
          mutation_prob, generatoin_limit,fitness_limit,top_sol,rent_ratio]
# Inputs[0] = new_weights
# Inputs[1] = new_profit
# Inputs[2] = node_weights
# Inputs[3] = weight_limit
# Inputs[4] = vmax
# Inputs[5] = vmin
# Inputs[6] = best_path
# Inputs[7] = item_per_city
# Inputs[8] = parent_size
# Inputs[9] = mutation_prob
# Inputs[10] = generatoin_limit
# Inputs[11] = fitness_limit
# Inputs[12] = top_sol
# Inputs[13] = rent_ratio


def generate_genome(length:int):
    # create random array of zeros and ones 
    # keeping last half zero in starting gives early convergence
    # since in prior we have less distance to travel
    return choices([0,1],k=int(length/2))+choices([0],k=length - int(length/2))

def generate_population(size:int, genome_length:int):
    # create population
    return [generate_genome(genome_length) for i in range(size)]

def fitness(genome,Inputs):
    
    w = 0   # increasing weight
    v = 0   # velocity while changing cities
    p = 0   # net profit (profit-rent)
    t = 0   # time when change the city
    
    # profit of items at current city
    loop_profit= np.zeros(len(Inputs[1]))
    
    for i in range(1,len(genome)+1):
        # weight addition one item at a time
        w += genome[i-1]*Inputs[0][i-1]
        if w <= Inputs[3]:
            
            # velocity = (vmax - (vmax-vmin)*(current weight/weight limit))
            v = Inputs[4] - (Inputs[4]-Inputs[5])*(w/Inputs[3])
            
            # checking when we jump the city 
            if (i)%Inputs[7]==0 and i<len(genome):
                
                # Calculate time for the jump (distance/velocity)
                t = np.abs(Inputs[2][i-1]-Inputs[2][i])/v
                
                # rent = rent ratio * time
                rent = Inputs[13]*t
                
                # we want to maximize the profit minimize the time
                # ~ mazimize (profit/time) and for that
                # ~ maximize ((profit/time)-rent)/weights
                
                #loop_profit[i:i+Inputs[7]] = (Inputs[1][i:i+Inputs[7]]-rent)/(t*Inputs[2][i:i+Inputs[7]])
                
                # or 
                # ~ maximize (profit-rent)/(t*weights)
                # converging faster
                loop_profit[i:i+Inputs[7]] = ((Inputs[1][i:i+Inputs[7]]/t)-rent)/Inputs[2][i:i+Inputs[7]]
                
            # adding all profit for the current city
            p += loop_profit[i-1]
            
        if w > Inputs[3]:
            p=0
            break
    return p
        


def selection_pair(population):
    # select top k parent to generate childs based on fitness
    new_population = choices(population=population,
                             weights=[fitness(genome,Inputs) for genome in population],
                             k = Inputs[8]
                             )
    return new_population


def single_point_crossover(a,b):
    
    #
    p = randint(1,len(a)-1)
    
    # checking feasibility to get childs
    if len(a)==len(b) and len(a)>2:
        return a,b
    
    # seperrating from random index and mixing two parents
    return a[0:p] + b[p:] , b[0:p] + a[p:]


def mutation(genome,Inputs):
    
    # Mutate based on mutation probability
    index = randrange(len(genome))
    if random()>Inputs[9]:
        genome[index] = genome[index]
    else:
        genome[index] = abs(genome[index]-1)
    return genome


def run_evolution(pop_size,genome_length,Inputs):
    prof= 0  # max profit at ith generation
    weg = 0 # weight according to profit at ith generation
    pop = generate_population(pop_size,genome_length)
    
    for i in range(Inputs[10]):
        if weg <= Inputs[3]:
            pop = sorted(pop, key=lambda genome: fitness(genome,Inputs),reverse=True)
            
            next_gen = pop[0:Inputs[12]]
            
            for j in range(int(len(pop)/2)-int(Inputs[12]/2)):
                parents = selection_pair(pop)
                child_a,child_b = single_point_crossover(parents[0],parents[1])
                child_a = mutation(child_a, Inputs)
                child_b = mutation(child_b, Inputs)
                next_gen += [child_a,child_b]
                
            pop = next_gen
            pop = sorted(pop, key=lambda genome: fitness(genome,Inputs),reverse=True)
            prof = np.sum(new_profit*pop[0])
            weg = np.sum(new_weights*pop[0])
            print("Generation = ",i,'\t',"Profit = ",prof,'\t',"Weight = ",weg)
        else:
            print("max profit = ",prof)
            break
        
    return pop[0]


pop = run_evolution(population_size, items, Inputs)
total_prof = np.sum(new_profit*pop)

# Plotting initialisation
fig, ax = plt.subplots()
ax.set(xlim=(min(x_loc)+5,max(x_loc)+5), ylim=(min(y_loc)+5,max(y_loc)+5))
ax.add_artist(plt.Circle((x_loc[0],y_loc[0]),1 , color='g'))

# re assigning profit gained from each city in the order of 0 to number cities
prof_city = np.zeros(dimension)
for i in range(int(len(pop)/item_per_city)):
    prof_city[best_path[i+1]] = np.sum(pop[i*item_per_city:(i*item_per_city)+item_per_city]*new_profit[i*item_per_city:(i*item_per_city)+item_per_city])
    x = x_loc[best_path[i+1]]
    y = y_loc[best_path[i+1]]
    r = 300 * prof_city[best_path[i+1]]/total_prof
    circ = plt.Circle((x,y),r, color='r')
    ax.add_artist(circ)

plt.show()        
   
# to plot TSP graph uncomment the line #380 from tsp.py 
# Not usefull
    
    