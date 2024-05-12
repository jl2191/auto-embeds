import logging
import os
from functools import wraps

from joblib import Memory

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
        its results and logs cache usage.
    """
    cachedir = os.path.join(os.getenv("AUTOEMBEDS_CACHE_DIR", "/tmp"), func.__name__)
    memory = Memory(cachedir, verbose=0)

    if os.getenv("AUTOEMBEDS_CACHING", "true").lower() == "true":
        cached_func = memory.cache(func)

        class CachedFunction:
            def __call__(self, *args, **kwargs):
                if memory.store_backend.contains_item([func, args, kwargs]):
                    logger.info(
                        f"Using cached result for {func.__name__} with args {args} and kwargs {kwargs}."
                    )
                    logger.info(
                        f"To clear the cache, call {func.__name__}.clear_cache()"
                    )
                return cached_func(*args, **kwargs)

            def clear_cache(self):
                logger.info(f"Clearing cache for {func.__name__}")
                memory.clear(warn=False)

        return CachedFunction()
    else:

        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        return wrapper
