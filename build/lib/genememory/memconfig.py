# TODO read all this from a file !

from genememory.constant import *

# global constants
TOP_P = "top_p"
BEAM_SEARCH = "beam_search"


class MemConfig(object):
    def __init__(self):
        self._const = GeneMemConstants()
        self._head_len = 0
        self._pop_size = 0
        self._load_path = ""
        self._save_path = ""
        self._symbol_set = self._const.syms
        self._gene_count = 1
        self._attention_head_count = 4
        self._num_dense_layer = 128
        self._embed_dim = 64
        self._tf_model_weights_path = "tmp_neural.sav"
        self._number_k_sampling = 4
        self._expression_count = 1
        self._decoding_method = TOP_P
        self._top_p_value = 0.95
        self._num_blocks = 4
        self._beam_width = 3
        self._beam_paths = 2
        self._selected_symbols = []
        self._feature_dim = 7  # mandatory parameter
        self._input_len = 256
        self._function_dct = [{}, {}]
        self._initial_training_dim = 3

    def set_initial_training_dim(self, val):
        self._initial_training_dim = val
        return self

    def set_function_dct(self, function_dct):
        self._function_dct = function_dct
        return self

    def set_input_len(self, input_len):
        self._input_len = input_len
        return self

    # will be droped after tokenizer is saved
    def set_selected_symbols(self, selected_symbols):
        self._selected_symbols = selected_symbols
        return self

    def set_feature_dim(self, value):
        self._feature_dim = value
        return self

    def set_beam_width(self, value):
        self._beam_width = value
        return self

    def set_beam_paths(self, value):
        self._beam_paths = value
        return self

    def set_num_blocks(self, value):
        self._num_blocks = value
        return self

    def set_top_p_value(self, value):
        self._top_p_value = value
        return self

    def set_decoding_method(self, method):
        self._decoding_method = method
        return self

    def set_number_expression_count(self, number):
        self._expression_count = number
        return self

    def set_number_k_sampling(self, number_k_sampling):
        self._number_k_sampling = number_k_sampling
        return self

    def set_tf_model_weights_path(self, tf_model_weights_path):
        self._tf_model_weights_path = tf_model_weights_path
        return self

    def set_embed_dim(self, dim):
        self._embed_dim = dim
        return self

    def set_attention_head_size(self, head_count):
        self._attention_head_count = head_count
        return self

    def set_num_dense_layer(self, num_dense_layer):
        self._num_dense_layer = num_dense_layer
        return self

    def set_gene_count(self, gene_count):
        self._gene_count = gene_count
        return self

    def head_len(self, len):
        self._head_len = len
        return self

    def set_pop_size(self, pop_size):
        self._pop_size = pop_size
        return self

    def set_load_path(self, load_path):
        self._load_path = load_path
        return self

    def set_save_path(self, save_path):
        self._save_path = save_path
        return self

    def set_symbol_set(self, symbols):
        self._symbol_set = symbols
        return self

    def get_head_len(self):
        return self._head_len

    def get_load_path(self):
        return self._load_path

    def get_save_path(self):
        return self._save_path

    def get_pop_size(self):
        return self._pop_size

    def get_symbol_set(self):
        return self._symbol_set

    def get_gene_count(self):
        return self._gene_count

    def get_attention_head_count(self):
        return self._attention_head_count

    def get_num_dense_layer(self):
        return self._num_dense_layer

    def get_embed_dim(self):
        return self._embed_dim

    def get_tf_model_weights_path(self):
        return self._tf_model_weights_path

    def get_number_k_sampling(self):
        return self._number_k_sampling

    def get_number_expression_count(self):
        return self._expression_count

    def get_decoding_method(self):
        return self._decoding_method

    def get_top_p_value(self):
        return self._top_p_value

    def get_num_blocks(self):
        return self._num_blocks

    def get_beam_width(self):
        return self._beam_width

    def get_beam_paths(self):
        return self._beam_paths

    def get_selected_symbols(self):
        return self._selected_symbols

    def get_feature_dim(self):
        return self._feature_dim

    def get_input_len(self):
        return self._input_len

    def get_function_dct(self):
        return self._function_dct

    def get_init_training_dim(self):
        return self._initial_training_dim