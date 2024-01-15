
import multiprocessing

def square_number(n):
    """Function to square a number."""
    return n * n

def main():
    # List of numbers to process
    numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    # Determine the number of processes
    num_processes = multiprocessing.cpu_count()

    # Create a pool of processes
    with multiprocessing.Pool(num_processes) as pool:
        # Apply 'square_number' to each element in 'numbers'
        squares = pool.map(square_number, numbers)

    # Print the result
    print("Squares:", squares)

if __name__ == '__main__':
    main()
















# import multiprocessing

# def my_function(args):
#     # Your function that does something
#     pass

# if __name__ == '__main__':
#     # Number of processes you want to spawn
#     num_processes = multiprocessing.cpu_count()  # Or set it to a fixed number
#     print(num_processes)

#     # Create a pool of processes
#     with multiprocessing.Pool(num_processes) as pool:
#         results = pool.map(my_function, iterable_of_arguments)

#     # 'results' contains the output of the function applied to each item 
#     # from 'iterable_of_arguments'







