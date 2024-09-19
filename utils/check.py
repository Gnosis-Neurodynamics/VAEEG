_all__ = ["check_value", "check_type"]


def check_value(parameter, value, allowed_values):
    """
    Check the value of a parameter against a list of valid opts.

    Return the value if it is valid, otherwise raise a ValueError with a
    readable error message.

    Args:
        parameter: str
            The name of the parameter to check. This is used in the error message.
        value: any type
            The value of the parameter to check.
        allowed_values: list
            The list of allowed values for the parameter.

    Raises:
        ValueError
            When the value of the parameter is not one of the valid opts.

    Returns:
        value : any type
            The value if it is valid.

    """
    check_type("parameter", parameter, [str])
    check_type("allowed_values", allowed_values, [list])

    if value in allowed_values:
        return value

    msg = ("Invalid value for the '{parameter}' parameter. "
           '{options}, but got {value!r} instead.')

    if len(allowed_values) == 0:
        raise RuntimeError("allowed_values is not set.")
    elif len(allowed_values) == 1:
        options = f'The only allowed value is {repr(allowed_values[0])}'
    else:
        options = 'Allowed values are '
        if len(allowed_values) == 2:
            options += ' and '.join(repr(v) for v in allowed_values)
        else:
            options += ', '.join(repr(v) for v in allowed_values[:-1])
            options += f', and {repr(allowed_values[-1])}'
    raise ValueError(msg.format(parameter=parameter, options=options,
                                value=value))


def check_type(parameter, value, allowed_types):
    """
    Check the value of a parameter against a list of valid type opts.

    Return the value if its type is valid, otherwise raise a TypeError with a
    readable error message.

    Args:
        parameter: str
            The name of the parameter to check. This is used in the error message.
        value: any type
            The value of the parameter to check.
        allowed_types: list
            The list of allowed types for the parameter.

    Raises:
        TypeError
            When the value of the parameter is not one of the valid type opts.

    Returns:
        value : any type
            The value if its type is valid.

    """
    if not isinstance(parameter, str):
        raise TypeError("`parameter` must be %s, but got %s" % (repr(str), repr(type(parameter))))

    if not isinstance(allowed_types, list):
        raise TypeError("`allowed_types` must be %s, but got %s" % (repr(list), repr(type(allowed_types))))

    allowed_types = tuple(allowed_types)

    if isinstance(value, allowed_types):
        return value

    msg = ("Invalid type for the '{parameter}' parameter: "
           '{options}, but got {value_type} instead.')

    if len(allowed_types) == 0:
        raise RuntimeError("allowed_types is not set.")
    elif len(allowed_types) == 1:
        options = f'The only allowed type is {repr(allowed_types[0])}'
    else:
        options = 'Allowed types are '
        if len(allowed_types) == 2:
            options += ' and '.join(repr(v) for v in allowed_types)
        else:
            options += ', '.join(repr(v) for v in allowed_types[:-1])
            options += f', and {repr(allowed_types[-1])}'
    raise TypeError(msg.format(parameter=parameter, options=options,
                               value_type=repr(type(value))))
