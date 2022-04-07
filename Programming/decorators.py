def decorator_f(original_f, *args, **kwargs):
    def wrapper_f(*args, **kwargs):
        print('Run this before {}'.format(original_f.__name__))
        return original_f(*args, **kwargs)

    return wrapper_f



def display():
    print('yo')


def display_info(name):
    print('Display with ({})'.format(name))

f = decorator_f(lambda: display_info(name='Luca'))
f()
quit()

quit()
display_info('Luca', 32)
