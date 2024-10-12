import numpy as np
from callback_utils import add_class_change_callbacks

class TestClass:
    def __init__(self, param_str, param_lst, param_dct, param_arr):
        self.param_str = param_str
        self.param_lst = param_lst
        self.param_dct = param_dct
        self.param_arr = param_arr

    def on_change(self, attr_name, attr_val):
        print(f"changing {attr_name} to {attr_val}")
        # the below is to check that callback is correctly disabled here
        # to prevent infinite recursion
        self.param_str = self.param_str

add_class_change_callbacks(TestClass)

if __name__ == "__main__":
    obj = TestClass(
        "abc",
        [1, 2, 3],
        {"foo": "bar", "hello": "world"},
        np.zeros(5))
    # shouldn't be getting any callbacks until the object is initialized

    # from here, touching anything should raise the callback
    obj.param_str = "xyz"
    obj.param_str += "abc"
    obj.param_lst[0] = -1
    obj.param_lst.append(None)
    del obj.param_lst[-1]
    obj.param_dct["foo"] = "something"
    obj.param_dct["hey"] = 3.14
    del obj.param_dct["hey"]
    obj.param_arr[0] = 1.0
    obj.param_arr[0] += 1.0
    obj.param_arr += -1.0
    obj.param_new = "something new"
