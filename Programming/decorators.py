def decorator_f(original_f):
    def wrapper_f(*args, **kwargs):
        print('Run this before {}'.format(original_f.__name__))
        return original_f(*args, **kwargs)

    return wrapper_f


@decorator_f
def display():
    print('yo')

@decorator_f
def display_info(name, age):
    print('Display with ({}, {})'.format(name, age))


display_info('Luca', 32)
