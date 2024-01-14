from dask import delayed
from distributed import Client, progress, wait

def main():
    client = Client()

    @delayed
    def add(x, y):
        import time
        time.sleep(1)
        return x + y

    @delayed
    def multiply(x, y):
        import time
        time.sleep(1)
        return x * y

    a = add(1, 2)
    b = add(3, 4)
    c = multiply(a, b)

    # result = c.compute(asyncio=True)
    future = client.compute(c)
    progress(future)
    # wait(future)

    # Wait for the result and print it
    print(type(future))

    result = future.result() # .result waits until the calculation is completed
    print(result)

if __name__ == '__main__':
    main()
