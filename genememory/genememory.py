import pickle
import genememory.nnutils.modelwrapper as mod
import random
from genememory.dto import TrainingRecord
from geppy.core.symbol import EphemeralTerminal
import numpy as np

class GeneMemory(object):
    """
        GenMemory represents a retention mechanism optimize genotype via learning
        just needs to be adapted to geppy
    """

    def __init__(self, mem_config, **kwargs):
        self._head_len = mem_config.get_head_len()
        self._load_path = mem_config.get_load_path()
        self._save_path = mem_config.get_save_path()
        self._symbol_set = mem_config.get_symbol_set()
        self._gene_length = self._head_len * 2 + 1
        self._tf_model_weights_path = mem_config.get_tf_model_weights_path()
        self._vocab = [sym.name for sym in self._symbol_set]
        self._arity_syms = list(filter(lambda x: x.arity > 0, self._symbol_set))
        self._model = mod.ModelWrapper(self._vocab, self._head_len,
                                       self._gene_length,
                                       embed_dim=mem_config.get_embed_dim(),
                                       feature_dim=mem_config.get_feature_dim(),
                                       num_heads=mem_config.get_attention_head_count(),
                                       num_dense_layer=mem_config.get_num_dense_layer(),
                                       arity_syms=self._arity_syms,
                                       exp_count=mem_config.get_number_expression_count(),
                                       sampling=mem_config.get_number_k_sampling(),
                                       model_path=self._tf_model_weights_path,
                                       selected_syms=mem_config.get_selected_symbols(),
                                       _decoding_method=mem_config.get_decoding_method(),
                                       _top_p_value=mem_config.get_top_p_value(),
                                       num_blocks=mem_config.get_num_blocks(),
                                       _beam_width=mem_config.get_beam_width(),
                                       _beam_paths=mem_config.get_beam_paths(),
                                       input_len=mem_config.get_input_len(),
                                       function_dct=mem_config.get_function_dct(), **kwargs)

    def update(self, population, **kwargs):
        error_vector = kwargs.get("error_vector", [1] * len(population))
        meta_embed = kwargs.get("meta_embed", [] * len(population))
        unit_maps = kwargs.get("unit_maps", [{}] * len(population))
        trainee = [TrainingRecord(raw_pheno=indi[0][0], linking_function=indi[0][1],
                                  meta_information=meta, unit_map=unit_map, error=error) for
                   indi, meta, unit_map, error in zip(population, meta_embed, unit_maps, error_vector)]
        random.shuffle(trainee)
        train_progress = self._model.train(trainee, **kwargs)

        return train_progress

    def save_tf_model(self, model_path=""):
        if len(model_path) == 0:
            model_path = self._tf_model_weights_path
        self._model.save_tf_model(model_path)

    def load_tf_model(self, model_path=""):
        model_path = self._tf_model_weights_path if len(model_path) == 0 else model_path
        self._model.load_tf_model(model_path)

    def get_load_path(self):
        return self._load_path

    def get_save_path(self):
        return self._save_path

    def update_model_params(self, **kwargs):
        self._model.update_params(**kwargs)

    #Todo fix that magic string thing :D
    def convert_to_symbol_object(self, symbol_as_string):
        for elem in self._symbol_set:
            if elem.name == symbol_as_string and elem.name != '?':
                return elem
            elif elem.name == '?':
                return EphemeralTerminal('?', gen=np.random.rand)

    def sample(self, **kwargs):
        state = kwargs['state']
        link_state = kwargs['state'][0]
        if len(link_state) == 0:
            kwargs['state'] = [[], []]
            kwargs['link_sampling'] = True
            raw_gene_mat = self._model.sample_gene_seq(**kwargs)[0][-1]
            links = [self.convert_to_symbol_object(i) for i in raw_gene_mat]
            return links
        else:
            kwargs['state'] = [self._convert_state_(state[0], link_processing=True),
                               self._convert_state_(state[1])]
            raw_gene_mat = self._model.sample_gene_seq(**kwargs)[-1][0][-1]
            raw_gene_mat = [self.convert_to_symbol_object(i) for i in raw_gene_mat]
        return raw_gene_mat

    def _convert_state_(self, x, link_processing=False):
        ret_val = []
        if x is None:
            return x
        if link_processing:
            ret_val = [str(elem.__name__) for elem in x]
        else:
            for gene in x:
                ret_val.append([str(elem.name) for elem in gene])
        return ret_val

    def get_memory(self):
        return self._model
