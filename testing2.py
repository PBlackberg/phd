import functools
import time
import random

print('executing')

def retry(exception_to_check, tries=4, delay=1, backoff=2, logger=None):
    def decorator_retry(func):
        @functools.wraps(func)
        def wrapper_retry(*args, **kwargs):
            mtries, mdelay = tries, delay
            while mtries > 1:
                try:
                    return func(*args, **kwargs)
                except exception_to_check as e:
                    print(f"Caught '{e}', retrying in {mdelay} seconds...")
                    time.sleep(mdelay)
                    mtries -= 1
                    mdelay *= backoff
            return func(*args, **kwargs)  # Last attempt
        return wrapper_retry
    return decorator_retry

@retry(ValueError, tries=5, delay=2, backoff=2)
def function_that_may_fail():
    # Simulate a task that has a chance to fail
    print("Attempting to perform a task...")
    if random.randint(0, 1) == 0:
        # Simulate a failure condition
        raise ValueError("Simulated task failure")
    else:
        print("Task succeeded!")

# Run the function to see the retry mechanism in action
function_that_may_fail()
