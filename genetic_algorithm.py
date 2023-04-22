import pygad
import numpy
import matplotlib.pyplot as plt
from os import system
import itertools
import math
import pandas as pd
from typing import Tuple
from sklearn.model_selection import KFold
from function_generator import function_generator
from sklearn.model_selection import train_test_split
from polyfit import calculate_error,load_data
MAX_POLYNOMIAL_DEGREE = 11
def create_GA_model(polynomial_degree:int):
    # Create an instance of the GA class inside the ga module. Some parameters are initialized within the constructor.
    ga_instance = pygad.GA(num_generations=500,
                           num_parents_mating=4, # Number of solutions to be selected as parents in the mating pool.
                           fitness_func=fitness_func,
                           sol_per_pop=8, # Number of solutions in the population.
                           num_genes=polynomial_degree+1,
                           init_range_low=1,
                           init_range_high=10,
                           mutation_percent_genes=30,
                           parent_selection_type="sss", # Type of parent selection.
                           keep_parents=1, # Number of parents to keep in the next population. -1 means keep all parents and 0 means keep nothing.
                           crossover_type="single_point",
                           mutation_type="random",
                           mutation_by_replacement=True,
                           random_mutation_min_val=-0.5,
                           random_mutation_max_val=0.5
                            )
    return ga_instance
X_train_cv=[]
y_train_cv=[]
def fitness_func(ga_instance, solution, solution_idx):
    # Calculate the fitness value by scoring on alll X_train_cv
    model = function_generator(solution)
    mse,standard_error = calculate_error(model,X_train_cv,y_train_cv)
    fitness = 1.0 /(mse+0.000001)
    return fitness 
def tran_ga():
    # Load data
    X,y = load_data()
    models_generated=0
    # Split data into test and train sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # for each degree of polynomial perfom k-fold cross validation
    degree_scores=[]
    best_solutions=[]
    for degree in range(1,MAX_POLYNOMIAL_DEGREE):
        cross_validation=KFold(n_splits=10,shuffle=True)
        degree_mse_stderr=[]
        best_solution_for_degree=None
        best_fitness_for_degree=0
        for train_index,test_index in cross_validation.split(X_train):
            global X_train_cv,y_train_cv
            X_train_cv,y_train_cv = X_train[train_index],y_train[train_index]
            X_test_cv,y_test_cv = X_train[test_index],y_train[test_index]
            
            ga_instance = create_GA_model(degree)
            ga_instance.run()
            solution, solution_fitness, solution_idx = ga_instance.best_solution()
            if solution_fitness>best_fitness_for_degree:
                best_fitness_for_degree=solution_fitness
                best_solution_for_degree=solution
            mse,standard_error = calculate_error(function_generator(solution),X_test_cv,y_test_cv)
            degree_mse_stderr.append((mse,standard_error))
        mean_mse = numpy.mean([x[0] for x in degree_mse_stderr])
        mean_stderr=numpy.mean([x[1] for x in degree_mse_stderr])
        degree_scores.append((mean_mse,mean_stderr))
        best_solutions.append(best_solution_for_degree)
        print(f"Degree {degree} has {mean_mse=} and {mean_stderr=}")
        print(f"Best solution for {degree=} is {best_solution_for_degree}")    
    #Plot results with labels
    plt.errorbar(x=[x for x in range(1,MAX_POLYNOMIAL_DEGREE)],y=[x[0] for x in degree_scores],yerr=[x[1] for x in degree_scores],label="Standard error")
    plt.legend()
    plt.xlabel("Polynomial degree")
    plt.ylabel("Mean squared error in m")
    plt.show()
    # Plot all models
    for degree in range(1,MAX_POLYNOMIAL_DEGREE):
        model = function_generator(best_solutions[degree-1])
        plt.scatter(X_test,model(X_test),label=f"Degree {degree}")
    plt.scatter(X_test,y_test,label="Real data")
    plt.xlabel("Distance from UWB")
    plt.ylabel("Corrected distance/Real distance")
    plt.legend()
    plt.show()
    
    # Pick the degree with the lowest mse
    best_degree = numpy.argmin([x[0] for x in degree_scores])
    print(f"Best degree is {best_degree+1}")
    # Get the lowest degree with mse within one standard error of the best degree
    best_degree_mse = degree_scores[best_degree][0]
    best_degree_stderr = degree_scores[best_degree][1]
    best_degree = numpy.argmin([x[0] for x in degree_scores if x[0] < best_degree_mse + best_degree_stderr])
    for i in range(MAX_POLYNOMIAL_DEGREE-1):
        if degree_scores[i][0] < best_degree_mse + best_degree_stderr:
            best_degree = i
            break
    best_degree=best_degree+1
    print(f"Best degree within standard error for the lowest mse is {best_degree}")
    
    # Create solution for the best degree
    X_train_cv,y_train_cv = X_train,y_train
    ga_instance = create_GA_model(best_degree)
    ga_instance.run()
    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    # Create model from solution
    model = function_generator(solution)
    mse,stardard_error = calculate_error(model,X_test,y_test)
    print(f"Best model has {mse=} and {stardard_error=}")
    # Plot best model
    plt.scatter(X_test,model(X_test),label=f"Degree {best_degree}")
    plt.scatter(X_test,y_test,label="Real data")
    plt.xlabel("Distance from UWB (m)")
    plt.ylabel("Corrected distance/Real distance (m)")
    plt.legend()
    plt.show()
    
  
                
if __name__ == "__main__":
    tran_ga()