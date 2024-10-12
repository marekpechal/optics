import numpy as np

class DictWithCallback(dict):
    def __setitem__(self, key, value):
        dict.__setitem__(self, key, value)
        self.on_change()
    def __delitem__(self, key):
        dict.__delitem__(self, key)
        self.on_change()

class ListWithCallback(list):
    def __setitem__(self, key, value):
        list.__setitem__(self, key, value)
        self.on_change()
    def __delitem__(self, key):
        list.__delitem__(self, key)
        self.on_change()
    def append(self, value):
        list.append(self, value)
        self.on_change()

class ArrayWithCallback(np.ndarray):
    def __setitem__(self, key, value):
        np.ndarray.__setitem__(self, key, value)
        self.on_change()

def add_class_change_callbacks(cls):
    # modify __setattr__ to keep track of changes
    setattr_orig = cls.__setattr__
    def setattr_mod(obj, attr_name, attr_val):
        # attach callback to attributes which can be modified in place
        if hasattr(obj, "on_change"):
            if isinstance(attr_val, dict):
                attr_val = DictWithCallback(attr_val.items())
                attr_val.on_change = lambda: obj.on_change(attr_name, attr_val)
            elif isinstance(attr_val, list):
                attr_val = ListWithCallback(attr_val)
                attr_val.on_change = lambda: obj.on_change(attr_name, attr_val)
            elif isinstance(attr_val, np.ndarray):
                attr_val = attr_val.view(ArrayWithCallback)
                attr_val.on_change = lambda: obj.on_change(attr_name, attr_val)

        setattr_orig(obj, attr_name, attr_val)

        if (hasattr(obj, "on_change") and obj._init_finished
                and attr_name != "_init_finished"):
            # execute callback if it's defined & __init__ is finished
            # ignore modification of the _init_finished flag itself
            obj.on_change(attr_name, attr_val)
    cls._init_finished = False
    cls.__setattr__ = setattr_mod

    # modify __init__ to raise a flag when finished
    init_orig = cls.__init__
    def init_mod(obj, *args, **kwargs):
        init_orig(obj, *args, **kwargs)
        obj._init_finished = True
    cls.__init__ = init_mod

    # modify on_change to disable callback inside
    if hasattr(cls, "on_change"):
        on_change_orig = cls.on_change
        def on_change_mod(obj, *args, **kwargs):
            obj._init_finished = False
            on_change_orig(obj, *args, **kwargs)
            obj._init_finished = True
        cls.on_change = on_change_mod
