import pygad
import numpy
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
cross_validation = KFold(n_splits=10)
max_polynomial_degree = 10
polynomial_scores = [0]*10
def calculate_error(model,inputs,expected)->tuple(float,float):
    assert len(inputs) == len(expected)
    errors=[]
    for i in range(len(inputs)):
        errors.append((model(inputs[i])-expected[i])/expected)
    errors_2 = [pow(x,2) for x in errors]
    mse=numpy.mean(errors_2)
    standard_error = numpy.sqrt(1/(len(errors_2)-1)*sum(errors_2))
    return (mse,standard_error)

def load_x():
    pass
def load_y():
    pass
models=[
    
]
def fitness_func(ga_instance, solution, solution_idx):
    pass
def main():
    # Load data
    X=load_x()
    y=load_y()
    
    # Split data into test and train sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    
    # Perform KFold Validation for ten polynomials
    for i in range(10):
        cross_validation=KFold(n_splits=10)
        for train_index, test_index in cross_validation.split(X_train):
            pass
    





function_inputs = [4,-2,3.5,5,-11,-4.7]
desired_output = 44
def fitness_func(ga_instance, solution, solution_idx):
    output = numpy.sum(solution*function_inputs)
    fitness = 1.0 / numpy.abs(output - desired_output)
    return fitness
fitness_function = fitness_func

num_generations = 50
num_parents_mating = 4

sol_per_pop = 8
num_genes = len(function_inputs)

init_range_low = -2
init_range_high = 5

parent_selection_type = "sss"
keep_parents = 1

crossover_type = "single_point"

mutation_type = "random"
mutation_percent_genes = 10
ga_instance = pygad.GA(num_generations=num_generations,
                       num_parents_mating=num_parents_mating,
                       fitness_func=fitness_function,
                       sol_per_pop=sol_per_pop,
                       num_genes=num_genes,
                       init_range_low=init_range_low,
                       init_range_high=init_range_high,
                       parent_selection_type=parent_selection_type,
                       keep_parents=keep_parents,
                       crossover_type=crossover_type,
                       mutation_type=mutation_type,
                       mutation_percent_genes=mutation_percent_genes)
ga_instance.run()
solution, solution_fitness, solution_idx = ga_instance.best_solution()
print("Parameters of the best solution : {solution}".format(solution=solution))
print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))

prediction = numpy.sum(numpy.array(function_inputs)*solution)
print("Predicted output based on the best solution : {prediction}".format(prediction=prediction))
print("works")