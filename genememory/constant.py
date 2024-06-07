import geppy as gep
import operator
import numpy as np


class GeneMemConstants(object):
    def __init__(self):
        self.token = {
            "add": '+',
            "mul": '*',
            "truediv": '/',
            'sub': "-"
        }

        def sqr(x):
            return x ** 2

        pset = gep.PrimitiveSet('Main', input_names=['x', 'x1', 'x2'])
        pset.add_function(operator.mul, 2)
        pset.add_function(operator.add, 2)
        pset.add_function(operator.sub, 2)
        pset.add_function(operator.truediv, 2)
        pset.add_function(np.cos, 1)
        pset.add_function(np.exp, 1)
        pset.add_function(np.log, 1)
        pset.add_function(np.sin, 1)
        pset.add_function(sqr, 1)
        pset.add_function(np.sqrt, 1)
        pset.add_function(np.tan, 1)
        pset.add_constant_terminal(-1)
        pset.add_constant_terminal(-.1)
        pset.add_constant_terminal(-.5)
        pset.add_constant_terminal(.5)
        pset.add_constant_terminal(.1)
        pset.add_constant_terminal(2)
        pset.add_constant_terminal(1)
        self.syms = [funcs for funcs in pset.functions] + [terms for terms in pset.terminals]
        for sym in self.syms:
            sym.nice_name = self.token.get(str(sym.name), str(sym.name))



