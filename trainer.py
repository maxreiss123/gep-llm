from genememory.memconfig import *
from genememory.constant import *
import genememory as geme
from geppy.support.simplification import *
import pandas as pd
import warnings
from sklearn.preprocessing import StandardScaler
from tests import *

warnings.filterwarnings("ignore")

"""
To train a retention mechanism we need to consider 4 steps:
1. Creating the vocabulary for the memory
2. Creating the language model with this memory
3. Load and split the data set 
4. Train the memory
"""


def sqr(x):
    return x ** 2


def create_dim_dictionary_for_case(temp):
    dim = eval(temp[0].replace('array', 'np.array'))
    return {f'x{i}': dim[i] for i in range(len(dim))}


def prepare_train_data_set(path_to_frame, source, max_dim=10, step_size=5):
    frame = pd.read_csv(path_to_frame, delimiter="&")
    frame['function_temp'] = list(map(lambda x: [x.split(',') for x in x.split(";")][::-1], frame['function']))
    frame_keys = frame['case'].unique()
    pop = []
    embed = []
    fits = []
    dimensions = []
    norm = dict()
    for key, value in source.items():
        norm[key] = StandardScaler().fit_transform(value[1])

    for elem in frame_keys:
        temp = frame[frame['case'] == elem]['function_temp'].to_list()
        temp = [[[elem[:-1], elem[-1]]] for elem in temp]
        temp_dims = frame[frame['case'] == elem]['dims'].to_list()

        # reshape_embed
        embed_temp = norm[elem]
        y_shape, x_shape = embed_temp.shape
        missing = np.abs(max_dim - x_shape)
        if missing > 0:
            x_pad = np.zeros((y_shape, missing))
            embed_temp = np.concatenate([embed_temp[:,:-1], x_pad, embed_temp[:, -1:]], axis=1)
            if step_size != 0:
                embed_temp_indices = np.random.randint(0, len(embed_temp), 2000)
                embed_temp = embed_temp[embed_temp_indices, :]
        dim = create_dim_dictionary_for_case(temp_dims)
        embed += [embed_temp] * len(temp)
        dimensions += [dim] * len(temp)
        pop += temp
        fits += frame[frame['case'] == elem]['fitness'].to_list()

    return pop, embed, fits, dimensions


def load_vocab():
    """
    vocab = [v1 for _, [v1, _, _, _] in source_task_dictionary.items()]
    vocab += [v1 for _, [v1, _, _, _] in target_tasks_dictionary.items()]
    vocab = list(set([item for sublist in vocab for item in sublist]))
    vocab = [elem.replace('lambda', 'lambda_') for elem in vocab]
    """
    vocab = [f'x{i}' for i in range(15)]

    pset = gep.PrimitiveSet('Main', input_names=vocab)
    # store it within a pset
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
    pset.add_function(np.power, 2)
    return pset


if __name__ == '__main__':
    # load vocab
    pset_voc = load_vocab()
    max_dim = 10
    # convert indis from the training dataset
    pop, embeds, err, dimensions \
        = prepare_train_data_set('training_data_set/new_eqs_very_small.csv', source_tasks, max_dim)

    # define the memory
    memconfig = MemConfig()
    memconstants = GeneMemConstants()
    number_place_holder = SymbolTerminal('?')
    syms = [funcs for funcs in pset_voc.functions] + [terms for terms in pset_voc.terminals] + [number_place_holder]
    for sym in syms:
        sym.nice_name = str(sym.name)
    embedding_size = 64
    (memconfig.head_len(5).set_gene_count(3).set_symbol_set(syms).set_embed_dim(embedding_size).
     set_attention_head_size(8).set_number_expression_count(1).
     set_num_blocks(6).set_feature_dim(max_dim).set_input_len(64).set_tf_model_weights_path("checkpoint.save"))
    gene_memory = geme.GeneMemory(memconfig)

    # trian the memory
    max_y_size = max([elem.shape[0] for elem in embeds])

    # need the value parameter
    print("Population size:", len(pop))
    pop_test = len(pop)//10
    batch_size = 64
    for epoch in range(1):
        hist = gene_memory.update(pop[:pop_test],
                                  train=True, epochs=100,
                                  error_vector=[1] * pop_test,
                                  meta_embed=embeds[:pop_test],
                                  batch_size=batch_size, frequency=10, max_y_size=max_y_size,
                                  test_size=0.3, unit_maps=dimensions[:pop_test]
                                  )
