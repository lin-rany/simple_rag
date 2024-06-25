import time

def timer_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = (end_time - start_time) * 1000  # convert to milliseconds
        print(f"Function {func.__name__} took {elapsed_time:.2f} ms to run.")
        return result
    return wrapper