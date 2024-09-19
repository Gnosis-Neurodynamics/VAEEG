import time
from functools import wraps
from inspect import signature

__all__ = ["timer", "type_assert"]


def timer(func):
    """Decorator to show the running time.

    Print the elapsed time when the decorated function run.
    """

    @wraps(func)
    def wrapper(*arg, **kw):
        start = time.time()
        r = func(*arg, **kw)
        end = time.time()
        total_time = end - start
        print("[%s] Elapsed time: %.5f s." % (func.__name__, total_time))
        return r

    return wrapper


def type_assert(*type_args, **type_kwargs):
    """
    Decorator for type checking.

    Args:
        *type_args:
        **type_kwargs:

    Returns:

    """

    def decorator(func):
        sig = signature(func)
        bound_types = sig.bind_partial(*type_args, **type_kwargs).arguments

        @wraps(func)
        def wrapper(*args, **kwargs):
            bound_values = sig.bind(*args, **kwargs)
            for name, value in bound_values.arguments.items():
                if name in bound_types:
                    if not isinstance(value, bound_types[name]):
                        raise TypeError('Argument {} must be {}'.format(name, bound_types[name]))
            return func(*args, **kwargs)

        return wrapper

    return decorator
