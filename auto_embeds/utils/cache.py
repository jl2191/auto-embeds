import os
from functools import wraps

from joblib import Memory

# Retrieve the cache directory from the environment variable or use '/tmp' as default.
cachedir = os.getenv("AUTOEMBEDS_CACHE_DIR", "/tmp")
memory = Memory(cachedir, verbose=0)


def auto_embeds_cache(func):
    """
    A decorator that caches the outputs of the decorated function in a directory
    named after the function itself. This caching mechanism is useful for functions
    with expensive or frequently repeated computations.

    The cache directory is determined by the environment variable `AUTOEMBEDS_CACHE_DIR`
    or defaults to '/tmp' if the variable is not set. Each function's cache is stored
    in a subdirectory under this path, named after the function.

    Usage:
        @auto_embeds_cache
        def expensive_function(param1, param2):
            # Function implementation
            return computation_result

        # Call the function, result is computed and cached
        result = expensive_function(1, 2)

        # Clear the cache for this specific function
        expensive_function.clear_cache()

    Attributes:
        None

    Returns:
        CachedFunction: A callable that behaves like the original function but caches
        its results.
    """

    cachedir = os.path.join(os.getenv("AUTOEMBEDS_CACHE_DIR", "/tmp"), func.__name__)
    memory = Memory(cachedir, verbose=0)

    if os.getenv("AUTOEMBEDS_CACHING", "true").lower() == "true":
        cached_func = memory.cache(func)

        class CachedFunction:
            def __call__(self, *args, **kwargs):
                return cached_func(*args, **kwargs)

            def clear_cache(self):
                memory.clear(warn=False)

        return CachedFunction()
    else:

        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        return wrapper
