import time
from functools import wraps

def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time

        # format time in hours, minutes, seconds
        hours, rem = divmod(total_time, 3600)
        minutes, seconds = divmod(rem, 60)

        time_str = "{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds)
        print(f'Function {func.__name__}{args} {kwargs} Took {time_str}')
        return result
    return timeit_wrapper