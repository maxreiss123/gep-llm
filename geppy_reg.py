import operator
import os.path
from geppy.core.symbol import EphemeralTerminal
import numpy as np
from deap import creator, base, tools
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from genememory.memconfig import *
from genememory.constant import *
import genememory as geme
from tests import *
import warnings
from geppy.support.simplification import *
import tensorflow as tf
import pandas as pd

warnings.filterwarnings("ignore")
import pickle as pkl
import random
from sklearn.preprocessing import StandardScaler
import geppy.tools.parser as ps
import numexpr


def sqr(x):
    return x ** 2


arity_map = {
    'power': 2,
    '+': 2,
    '-': 2,
    '*': 2,
    '/': 2,
    'log': 1,
    '**2': -1,
    '**(1/2)': -1,
    'exp': 1,
    'sin': 1,
    'cos': 1,
    'abs': 1,
}

function_map = {
    operator.add.__name__: '+',
    operator.sub.__name__: '-',
    operator.mul.__name__: '*',
    np.power.__name__: '**',
    operator.abs.__name__: 'abs',
    operator.truediv.__name__: '/',
    math.log.__name__: 'log',
    math.sin.__name__: 'sin',
    math.cos.__name__: 'cos',
    np.exp.__name__: 'exp',
    np.sqrt.__name__: '**(1/2)',
    'sqr': '**2',
    '//': '/'
}

CONSTANTS_NAT = {
    'GC_': 6.67430e-11,
    'CGA_': 9.80665,
    'CSL_': 2.99792458e8,
    'CE_': 8.854e-12,
    'CPL_': 6.626e-34,
    'CBO_': 1.380649e-23,
    'CBM_': 9.2740100783e-24,
    'CDIR_': 1.054571817e-34,
    'CEMA_': 9.10938356e-31,
    'CFS_': 7.2973525693e-3
}

MATH_FUNC_MAP = {
    '+': [operator.add, 2],
    '-': [operator.sub, 2],
    '*': [operator.mul, 2],
    '/': [operator.truediv, 2],
    'sqr': [sqr, 1],
    'sin': [np.sin, 1],
    'cos': [np.cos, 1],
    'sqrt': [np.sqrt, 1],
    'exp': [np.exp, 1],
    'log': [np.log, 1]
}

GEPPY_FUN_MAP = {
    'add': operator.add,
    'sub': operator.sub,
    'mul': operator.mul,
    'truediv': operator.truediv,
    'sqr': sqr,
    'sin': np.sin,
    'cos': np.cos,
    'sqrt': np.sqrt,
    'exp': np.exp,
    'log': np.log
}
fun_dict = {}

def evaluate(individual, toolbox, data_set):
    """Evalute the fitness of an individual: MAE (mean absolute error)"""
    func = toolbox.compile(individual)
    if func in fun_dict:
        return fun_dict[func],
    try:
        # data = StandardScaler().fit_transform(data_set.values)
        # data = pd.DataFrame(data, columns=data_set.columns)
        Yp = numexpr.evaluate(func, data_set)
        Yt = data_set['f']
        err = mean_absolute_error(Yt, Yp)
        if err < 0:
            err = 10e5
    except Exception as error_msg:
        err = 10e5
    fun_dict[func] = err
    return err,


def log_genes(population, gen, file_name, seed, symbol_table=None, case=''):
    records = []
    for indi in population:
        record = {'gen': gen, 'seed': seed, 'fitness': indi.fitness.values,
                  'function': indi.print_raw_pheno_as_string(symbol_table), 'case': case}
        records.append(record)
    frame = pd.DataFrame(records)
    #frame.to_csv(file_name, header=not os.path.exists(file_name), sep=';', mode='a')


def run(seed, data_set, case_name, file_name='', pop_size_=20, ngens_=250, prop=0.9, mem=False, **kwargs):
    # data_set = pd.DataFrame(StandardScaler().fit_transform(data_set.values))
    data = {prim: data_set.values[:, i] for prim, i in zip(kwargs.get('primitives', ['x']) + ['f'],
                                                           range(len(kwargs.get('primitives', ['x'])) + 1))}
    tf.random.set_seed(0)
    np.random.seed(seed)
    random.seed(seed)
    prop_mem = kwargs.get('prop_mem', 0)

    def log_per_epoch(log, min_uuid, gen):
        if gen%5==0:
            entry = log[-1]
            entry['min_uuid'] = min_uuid
            entry['seed'] = seed
            entry['case_name'] = case_name
            entry['mem_usage'] = mem
            entry['prop_mem'] = prop_mem
            if os.path.exists(file_name):
                pd.DataFrame([entry]).to_csv(file_name, mode='a', header=False)
            else:
                pd.DataFrame([entry]).to_csv(file_name, mode='a')

    random.seed(seed)
    np.random.seed(seed)
    pset = gep.PrimitiveSet('Main', input_names=kwargs.get('primitives', ['x']))

    if len(kwargs.get('func_', [])) > 0:
        funcs = kwargs.get('func_')
        funcs = [MATH_FUNC_MAP.get(elem, None) for elem in funcs]
        for elem in funcs:
            if elem is not None:
                pset.add_function(elem[0], elem[1])

    else:
        pset.add_function(operator.mul, 2)
        pset.add_function(operator.add, 2)
        pset.add_function(operator.sub, 2)
        pset.add_function(operator.truediv, 2)
        pset.add_function(sqr, 1)
        pset.add_function(np.sqrt, 1)
        #pset.add_function(np.cos, 1)
        #pset.add_function(np.exp, 1)
        #pset.add_function(np.log, 1)
        #pset.add_function(np.sin, 1)
        #pset.add_function(np.tan, 1)
        #pset.add_function(np.power, 2)

    creator.create("FitnessMin", base.Fitness, weights=(-1,))
    creator.create("Individual", gep.Chromosome, fitness=creator.FitnessMin)

    h = kwargs.get('head_size', 6)  # head length
    n_genes = kwargs.get('n_egenes', 3)  # number of genes in a chromosome
    target_dim = 10
    r = 4
    # toolbox.register('rnc_gen', np.random.randint,low=-10, high=10)

    toolbox = gep.Toolbox()

    toolbox.register('write_pop', log_genes, file_name=kwargs.get('log_name', 'genes.csv'), seed=seed,
                     case=case_name)
    pset.add_ephemeral_terminal(name='?', gen=np.random.rand)
    if mem:
        memconfig = MemConfig()
        handler = open('vocab.pkl', 'rb')
        stand_syms = pkl.load(handler)
        handler.close()

        syms = [funcs for funcs in pset.functions] + [SymbolTerminal(f'x{i}') for i in
                                                      range(len(kwargs.get('primitives', ['x'])))] +\
               [EphemeralTerminal('?', np.random.random)]
        for sym in stand_syms:
            sym.nice_name = str(sym.name)
        (memconfig.head_len(h).set_gene_count(n_genes).set_number_expression_count(1).
         set_tf_model_weights_path("checkpoint.save").set_symbol_set(stand_syms).set_feature_dim(target_dim).
         set_selected_symbols(syms).set_top_p_value(0.95).set_input_len(64))
        gene_memory = geme.GeneMemory(memconfig, physical_units=kwargs.get('unit_maps', {}))
        embedding = StandardScaler().fit_transform(data_set)
        embedding = np.concatenate([embedding[:, :-1], np.zeros((embedding.shape[0],
                                                                 np.abs((embedding.shape[1]) - target_dim)))[:, :],
                                    embedding[:, -1:]], axis=1)
        toolbox.register('gene_gen', gep.GeneLM, pset=pset, head_length=h, gene_memory=gene_memory,
                         embedding=embedding, temperature=1.2, top_k=1)
    else:

        toolbox.register('gene_gen', gep.Gene, pset=pset, head_length=h)
    toolbox.register('individual', creator.Individual, gene_gen=toolbox.gene_gen, n_genes=n_genes,
                     linker=[operator.add,operator.mul, operator.truediv, operator.sub],
                     conditional=mem, prop_threshold=kwargs.get('prop_mem', 0),
                     function_map=GEPPY_FUN_MAP)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register('compile', ps.compile_to_function_string, pset=pset, arity_map=arity_map,
                     function_map=function_map)

    toolbox.register('log_per_epoch', log_per_epoch)
    toolbox.register('evaluate', evaluate, data_set=data, toolbox=toolbox)
    toolbox.register('select', tools.selTournament, tournsize=2)
    toolbox.register('mut_uniform', gep.mutate_uniform, pset=pset, ind_pb=0.05, pb=1)
    #toolbox.register('mut_invert', gep.invert, pb=0.2)
    toolbox.register('mut_is_transpose', gep.is_transpose, pb=0.1)
    toolbox.register('mut_ris_transpose', gep.ris_transpose, pb=0.1)
    #toolbox.register('mut_gene_transpose', gep.gene_transpose, pb=0.1)
    toolbox.register('cx_1p', gep.crossover_one_point, pb=0.3)
    toolbox.register('cx_2p', gep.crossover_two_point, pb=0.2)
    #toolbox.register('cx_gene', gep.crossover_gene, pb=0.05)

    stats = tools.Statistics(key=lambda ind: ind.fitness.values[0])
    stats.register("avg", np.nanmean)
    stats.register("std", np.nanstd)
    stats.register("min", np.nanmin)
    stats.register("max", np.nanmax)

    # size of population and number of generations
    n_pop = pop_size_
    n_gen = ngens_

    pop = toolbox.population(n=n_pop)
    hof = tools.HallOfFame(3)

    pop, log = gep.gep_simple(pop, toolbox, n_generations=n_gen, n_elites=int(pop_size_ * (1 - prop)),
                              stats=stats, hall_of_fame=hof, verbose=False, epsilon=1e-4,
                              log_genes=kwargs.get('log_genes', False), train_mem=False)
    return log
