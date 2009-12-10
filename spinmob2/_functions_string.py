


def join(array_of_strings, delimiter=' '):
    if array_of_strings == []: return ""

    output = str(array_of_strings[0])
    for n in range(1, len(array_of_strings)):
        output += delimiter + str(array_of_strings[n])
    return(output)

def ubersplit(s, delimiters=['\t','\r',' ']):

    # run through the string, replacing all the delimiters with the first delimiter
    for d in delimiters: s = s.replace(d, delimiters[0])
    return s.split(delimiters[0])
