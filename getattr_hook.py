from types import ModuleType
stuff = {}  # type: ignore
def log(key, value):  # type: ignore
    if isinstance(key, ModuleType):
        key = (str(type(key)), key.__name__)
    else:
        key = str(type(key))
    idx = (key, value)
    stuff[idx] = stuff.get(idx, 0) + 1
def log_method(key, value):  # type: ignore
    log(key, value)
