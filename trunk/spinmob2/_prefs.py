import os         as _os
import _functions as _fun


class Prefs():

    def __init__(self, parent):
        """
        This class holds all the user-variables, paths etc...

        You should manually call load() after creating this.
        """

        self.parent = parent
        self.prefs  = {}
        self.prefs_keys = []

        # set up the variables based on operating system
        if not _os.environ.has_key("HOME"):
            _os.environ['HOME'] = _os.environ['USERPROFILE']
        self.path_delimiter = _os.path.sep

        # assemble the home and temp directory path for this environment
        self.home_dir      = _os.environ['HOME'] + self.path_delimiter + '.spinmob2'
        self.temp_dir      = self.home_dir       + self.path_delimiter + 'temp'
        self.prefs_path    = self.home_dir       + self.path_delimiter + 'preferences.txt'
        self.colormaps_dir = self.home_dir       + self.path_delimiter + 'colormaps'

        # see if this is the first time running (no home directory)
        if not _os.path.exists(self.home_dir):
            print "Creating "+self.home_dir
            _os.mkdir(self.home_dir)

        if not _os.path.exists(self.temp_dir):
            print "Creating "+self.temp_dir
            _os.mkdir(self.temp_dir)

        if not _os.path.exists(self.prefs_path):
            print "Creating "+self.prefs_path
            open(self.prefs_path, 'w').close()

        if not _os.path.exists(self.colormaps_dir):
            print "Creating "+self.colormaps_dir
            _os.mkdir(self.colormaps_dir)

        # now read in the prefs file
        lines = _fun.file.read_lines(self.prefs_path)
        self.prefs = {}
        for n in range(0,len(lines)):
            s = lines[n].split('=')
            if len(s) > 1:
                self.prefs[s[0].strip()] = s[1].strip()


    def load(self, name, clear=True):
        """
        Loads the prefs file name.cfg
        """

        # assemble the home and temp directory path for this environment
        self.prefs_path = name + '.cfg'

        # if we don't have a prefs file, create one!
        if not _os.path.exists(self.prefs_path):
            print "Creating "+self.prefs_path + "..."
            open(self.prefs_path, 'w').close()

        # now read in the prefs file
        lines = _fun.file.read_lines(self.prefs_path)

        # now parse the lines
        if clear: self.prefs = {}
        for n in range(0,len(lines)):
            s = lines[n].split('=')

            # if this line is valid, len(s) > 1
            if len(s) > 1:
                k = s[0].strip()
                v = s[1].strip()

                # first try evaluating it, then assume it's a string
                try:
                    a = eval(v)
                    self.prefs[k] = a
                except:
                    self.prefs[k] = v

                # if we haven't seen this key yet, append it to the key list (for ordering)
                if not k in self.prefs_keys: self.prefs_keys.append(k)

        # if we didn't clear it, it means we're updating with a second file.
        # dump the whole thing in this case!
        if not clear: self.Dump()

    # function allowing us to type prefs("whatever")
    def __call__   (self, key): return self.Get(key)

    # function allowing us to type prefs["whatever"]
    def __getitem__(self,key):  return self.Get(key)

    # function allowing us to type prefs["whatever"] = 32
    def __setitem__(self,key,value): self.Set(key, value)

    # returns str(prefs)
    def __str__(self):
        s = ''
        for key in self.prefs.keys():
            s = s + key + " = " + self.prefs[key] + '\n'
        return s

    def keys(self):         return self.prefs.keys()
    def has_key(self, key): return self.prefs.has_key(key)

    def Get(self, key):
        """
        Checks if the key exists and returns it. Returns None if it doesn't
        """
        if self.prefs.has_key(key):
            return self.prefs[key]
        else:
            return None

    def Set(self, key, value):
        """
        Sets the key-value pair and dumps to the preferences file.
        """
        if not value == None:
            self.prefs[key] = value
            if not key in self.prefs_keys: self.prefs_keys.append(key)

        else:
            self.prefs.pop(key)
            self.prefs_keys.remove(key)

        self.Dump()

    def Remove(self, key):
        """
        Removes a key/value pair
        """
        self.Set(key, None)

    def Dump(self):
        """
        Dumps the current prefs to the preferences.txt file
        """
        prefs_file = open(self.prefs_path, 'w')
        for k in self.prefs_keys:
            prefs_file.write(str(k) + ' = ' + str(self.prefs[k]) + '\n')
        prefs_file.close()


