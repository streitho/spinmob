from _functions_string import join


def write_to_file(path, string):
    file = open(path, 'w')
    file.write(string)
    file.close()

def append_to_file(path, string):
    file = open(path, 'a')
    file.write(string)
    file.close()

def read_lines(path):
    file = open(path, 'r')
    a = file.readlines()
    file.close()
    return(join(a,'').replace("\r\n", "\n").replace("\r","\n").split("\n"))



def save_object(object, path="ask", text="Save this object where?"):
    if path=="ask": path = _dialogs.Save("*.pickle", text=text)
    if path == "": return

    if len(path.split(".")) <= 1 or not path.split(".")[-1] == "pickle":
        path = path + ".pickle"

    object._path = path

    f = open(path, "w")
    _cPickle.dump(object, f)
    f.close()

def load_object(path="ask", text="Load a pickled object."):
    if path=="ask": path = _dialogs.SingleFile("*.pickle", text=text)
    if path == "": return None

    f = open(path, "r")
    object = _cPickle.load(f)
    f.close()

    object._path = path
    return object

def replace_lines_in_files(search_string, replacement_line):

    # have the user select some files
    paths = _dialog.MultipleFiles('DIS AND DAT|*.*')
    if paths == []: return

    for path in paths:
        shutil.copy(path, path+".backup")
        lines = _fun.read_lines(path)
        for n in range(0,len(lines)):
            if lines[n].find(search_string) >= 0:
                print lines[n]
                lines[n] = replacement_line.strip() + "\n"
        _fun.write_to_file(path, _fun.join(lines, ''))

    return

def search_and_replace_in_files(search, replace, depth=100, paths="ask", confirm=True):

    # have the user select some files
    if paths=="ask":
        paths = _dialog.MultipleFiles('DIS AND DAT|*.*')
    if paths == []: return

    for path in paths:
        lines = _fun.read_lines(path)

        if depth: N=min(len(lines),depth)
        else:     N=len(lines)

        for n in range(0,N):
            if lines[n].find(search) >= 0:
                lines[n] = lines[n].replace(search,replace).strip()
                print path.split('\\')[-1]+ ': "'+lines[n]+'"'
                _wx.Yield()

        # only write if we're not confirming
        if not confirm: _fun.write_to_file(path, _fun.join(lines, '\n'))

    if confirm:
        if raw_input("ja? ")=="yes":
            search_and_replace_in_files(search,replace,depth,paths,False)

    return
