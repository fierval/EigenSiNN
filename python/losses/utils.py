def to_cpp(x):

    s = str(x).replace("[", "{").replace("]", "}").replace("tensor", "")
    return s