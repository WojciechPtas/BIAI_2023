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
MAX_POLYNOMIAL_DEGREE = 11
def calculate_error(model,inputs,expected):
    assert len(inputs) == len(expected)
    errors=[[],[]]
    for (x,y) in zip(inputs,expected):
        diff = model(x)-y
        errors[0].append(diff)
        errors[1].append(diff**2)
    mse = numpy.mean(errors[1])
    standard_error = numpy.std(errors[0])
    return (mse,standard_error)

def load_data():
    data = pd.read_csv('data.csv')
    X = data.iloc[:,0].values
    y = data.iloc[:,1].values
    return X,y

def train_polyfit():
    # Load data
    X,y = load_data()
    # Split data into test and train sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    degree_scores=[]
    best_solutions=[]
    for degree in range(1,MAX_POLYNOMIAL_DEGREE):
        cross_validation=KFold(n_splits=10,shuffle=True)
        degree_mse_stderr=[]
        best_solution_for_degree=None
        best_solution_score=0
        # for each degree of polynomial perfom k-fold cross validation
        for train_index,test_index in cross_validation.split(X_train):
            X_train_cv,y_train_cv = X_train[train_index],y_train[train_index]
            X_test_cv,y_test_cv = X_train[test_index],y_train[test_index]
            # Create polynomial model with numpy.polyfitMAX_POLYNOMIAL_DEGREE
            solution = numpy.polyfit(X_train_cv,y_train_cv,degree)
            solution = numpy.flip(solution)
            # Create model from solution
            model = function_generator(solution)
            mse,stardard_error = calculate_error(model,X_test_cv,y_test_cv)
            model_score = 1.0 / (mse+0.000001)
            if model_score > best_solution_score:
                best_solution_score = model_score
                best_solution_for_degree = solution
            degree_mse_stderr.append((mse,stardard_error))
        # Calculate average error for degree
        mean_mse = numpy.mean([x[0] for x in degree_mse_stderr])
        mean_stderr = numpy.mean([x[1] for x in degree_mse_stderr])/math.sqrt(len(degree_mse_stderr))
        degree_scores.append((mean_mse,mean_stderr))
        best_solutions.append(best_solution_for_degree)    
        print(f"Degree {degree} has {mean_mse=} and {mean_stderr=}")
        print(f"Best solution for {degree=} is {best_solution_for_degree}")    
    #Plot results with labels
    plt.errorbar(x=[x for x in range(1,MAX_POLYNOMIAL_DEGREE)],y=[x[0] for x in degree_scores],yerr=[x[1] for x in degree_scores],label="Standard error")
    plt.legend()
    plt.show()
    # Plot all models
    for degree in range(1,MAX_POLYNOMIAL_DEGREE):
        model = function_generator(best_solutions[degree-1])
        plt.scatter(X_test,model(X_test),label=f"Degree {degree}")
    plt.scatter(X_test,y_test,label="Real data")
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
    
    # Create model for the best degree
    solution = numpy.polyfit(X_train,y_train,best_degree)
    solution = numpy.flip(solution)
    # Create model from solution
    model = function_generator(solution)
    mse,stardard_error = calculate_error(model,X_test,y_test)
    print(f"Best model has {mse=} and {stardard_error=}")
    # Plot best model
    plt.scatter(X_test,model(X_test),label=f"Degree {best_degree}")
    plt.scatter(X_test,y_test,label="Real data")
    plt.legend()
    plt.show()
    
  
            
            
            
            
            
            
            
    #         #create and train ten GA models and than get the best solution from them
    #         best_solution = None
    #         best_fitness = 0
    #         # for i in range(10):
    #         #     ga_instance = create_GA_model(degree)
    #         #     ga_instance.run()
    #         #     #models_generated+=1
    #         #     #system("cls")
    #         #     #print(f"We already have generated {models_generated} models")
    #         #     if ga_instance.best_solution()[1] > best_fitness:
    #         #         best_fitness = ga_instance.best_solution()[1]
    #         #         best_solution = ga_instance.best_solution()[0]
    #         # # Create model from best solution
    #         # if best_fitness_for_degree < best_fitness:
    #         #     best_fitness_for_degree = best_fitness
    #         #     best_solution_for_degree = best_solution
            
    #         #create a polynomial model with numpy
    #         best_solution = numpy.polyfit(X_train_cv,y_train_cv,degree)
    #         best_solution = numpy.flip(best_solution)
    #         #print(best_solution)
    #         # Create model from best solution         
    #         model = function_generator(best_solution)
    #         # Calculate error
    #         mse,standard_error = calculate_error(model,X_test_cv,y_test_cv)
    #         # Add error to degree_scores
    #         degree_values.append((degree,mse,standard_error))
    #     print(f"Best solution for degree {degree} is {best_solution}")
    #     # Calculate mean error for degree
    #     degree_scores.append((degree,numpy.mean([x[1] for x in degree_values]),numpy.mean([x[2] for x in degree_values])))
    # # Scatter degree_scores and their standard error with matplotlib
    # df = pd.DataFrame(degree_scores,columns=['degree','mse','standard_error'])
    # print(df)
if __name__ == "__main__":
    train_polyfit()