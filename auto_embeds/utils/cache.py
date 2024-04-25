import os
from functools import wraps

from joblib import Memory

# Retrieve the cache directory from the environment variable or use '/tmp' as default.
cachedir = os.getenv("AUTOEMBEDS_CACHE_DIR", "/tmp")
memory = Memory(cachedir, verbose=0)


def auto_embeds_cache(func):
    """
    A decorator that caches the output of a function using joblib.Memory.
    Caching is controlled by the environment variable 'AUTOEMBEDS_CACHING'.
    If 'AUTOEMBEDS_CACHING' is set to 'True', caching is enabled.
    If 'AUTOEMBEDS_CACHING' is set to 'False' or not set, caching is disabled.

    Args:
        func (callable): The function to be potentially cached.

    Returns:
        callable: The cached function if caching is enabled, otherwise the original
            function.
    """
    if os.getenv("AUTOEMBEDS_CACHING", "true").lower() == "true":
        return memory.cache(func)
    else:
        # Return the original function without caching.
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        return wrapper
