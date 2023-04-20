def function_generator(coefficent_list: list[float]):
    def function(x: float) -> float:
        result = 0
        for i in range(len(coefficent_list)):
            result += coefficent_list[i] * (x ** i)
        return result
    return function

if __name__ == "__main__":
    # Path: main.py
    from function_generator import function_generator
    f = function_generator([1, 2, 3, 4])
    print(f(1))
    print(f(2))
    print(f(3))
    print(f(4))