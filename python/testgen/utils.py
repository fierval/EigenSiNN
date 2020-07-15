def to_cpp(x):
    return str(x).replace("[", "{").replace("]", "}")