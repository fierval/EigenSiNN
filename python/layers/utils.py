import re

def to_cpp(x):

    s = str(x).replace("[", "{").replace("]", "}").replace("tensor", "")

    # remove any other tensor parameters at the end
    regex = r"\,\s+[A-z]+.+\)$"
    s = re.sub(regex, ')', s)

    return s