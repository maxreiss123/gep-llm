import tensorflow as tf
import numpy as np
import genememory.nnutils.transformer as tfr
from tensorflow import keras
from os.path import exists
import itertools
import official.nlp.modeling.ops.sampling_module as smp
import genememory.memconfig as memco
from genememory.dto import TrainingRecord


class ModelWrapper(object):
    link_t_name_o = "link_o"
    gene_t_name_o = "gene_o"
    link_t_name_c = "link_c"
    gene_t_name_c = "gene_c"
    padding_sign = "padding_sign"

    # subs by kwargs
    def __init__(self, vocab, head_len, gen_size, embed_dim, num_heads, arity_syms,
                 exp_count, sampling, model_path, feature_dim, **kwargs):
        self._decoding_method = memco.TOP_P
        self._top_p_value = 0.8
        self._feature_dim = feature_dim
        self._num_blocks = kwargs.get("num_blocks", 4)
        self._selected_syms = kwargs.get("selected_syms")
        self._orig_vocab_list = vocab
        self._num_heads = num_heads
        self._max_terms = kwargs.get('max_terms', 1)
        self._embed_dim = embed_dim
        self._head_len = head_len
        self._max_complexity = kwargs.get('complexity_max', self._head_len * 1.5)
        self._temperature_numbers = 0.3
        self._exp_count = exp_count
        self._arity_syms_dict = {k: v for k, v in zip([a.nice_name for a in arity_syms], [b.arity for b in arity_syms])}
        self._chromosome_positions = self.create_h_token()
        self._link_token = {ModelWrapper.link_t_name_c: ["</LINK>"], ModelWrapper.gene_t_name_c: ["</GENE>"]}
        self._vocab = self._chromosome_positions + list(i[0] for _, i in self._link_token.items()) + \
                      sorted(set([k for k, v in self._arity_syms_dict.items() if v > 1])) + \
                      sorted(set([k for k, v in self._arity_syms_dict.items() if v < 2])) + \
                      sorted(set(vocab).symmetric_difference(set(self._arity_syms_dict.keys())))
        self._vocab_size = len(self._vocab) + 1

        self._last_train_hist = None
        self._arity_syms = arity_syms
        self._char2int_ = tf.keras.layers.StringLookup(vocabulary=list(self._vocab), mask_token=None)
        self._int2char_ = tf.keras.layers.StringLookup(vocabulary=self._char2int_.get_vocabulary(), invert=True,
                                                       mask_token=None)

        self._physical_units = {self._char2int_(x).numpy(): value for x, value in
                                kwargs.get('physical_units', {}).items()}
        self._pad_array = []
        self._train_samples = []
        self._gene_size = gen_size
        self._input_len = kwargs.get("input_len", 256)
        self._depend_state = kwargs.get("depend_state", self._input_len == 256)
        self._number_k_sampling = sampling
        self._beam_paths = 1
        self._beam_width = 1

        self.start_lr = 4e-5
        self.min_lr = 1e-4
        self.max_lr = 5e-3
        self.rampup_epochs = 10
        self.sustain_epochs = 2
        self.exp_decay = .8

        self._case_inference_vec = tf.constant(np.zeros((1, self._input_len)), dtype=tf.float32)
        self.__dict__.update(kwargs)
        self._prior_mask, self._connector_prior_mask \
            = self._create_tensor_mask(extension=self._max_terms + len(self._link_token) + 1,
                                       link_size=len([a for a in self._arity_syms if a.arity > 1]))
        self._model = self.load_tf_model(model_path, num_blocks=self._num_blocks)

    def create_h_token(self):
        return [f"<{number}>" for number in range(self._max_terms)]

    def _create_tensor_mask(self, extension, link_size):
        print(self._char2int_.get_vocabulary())

        all_possible = np.zeros(len(self._char2int_.get_vocabulary()))
        all_possible[:extension] = -np.inf
        only_terminals = np.copy(all_possible)
        only_terminals[:extension + len(self._arity_syms)] = -np.inf

        if len(self._selected_syms) > 0:
            target_indices = list(set(range(len(self._char2int_.get_vocabulary()))). \
                                  difference(set([self._char2int_(i.name).numpy() for i in self._selected_syms])))
            only_terminals[target_indices] = -np.inf
            all_possible[target_indices] = -np.inf
        self._pad_array = only_terminals
        gene_prob_mask = [tf.constant(all_possible, dtype=tf.float32) for _ in range(self._head_len)] + \
                         [tf.constant(only_terminals, dtype=tf.float32) for _ in
                          range(self._gene_size - self._head_len)]

        connector_prob_mask = np.array([-np.inf for _ in range(len(all_possible))])

        if len(self._selected_syms) > 0:
            names_ = [elem.name for elem in self._selected_syms]
            target_indices = list(set(range(len(self._char2int_.get_vocabulary()))). \
                                  intersection(set([self._char2int_(name).numpy() for name, arity in
                                                    self._arity_syms_dict.items() if arity > 1 and name in names_])))
            connector_prob_mask[target_indices] = 0
        else:
            connector_prob_mask[extension:extension + link_size] = 0

        connector_prob_mask = [tf.convert_to_tensor(connector_prob_mask, dtype=tf.float32)]

        return gene_prob_mask, connector_prob_mask

    # TODO bring to data obj.
    def set_number_k_sampling(self, number):
        self._number_k_sampling = number
        return self

    def get_model(self):
        return self._model

    def load_tf_model(self, path, num_blocks):
        if exists(path):
            model = keras.models.load_model(path)
            print(f"Load memory: {path}")
        else:
            model = tfr.create_model(max_len=self._input_len, vocab_size=self._vocab_size,
                                     embed_dim=self._embed_dim,
                                     num_heads=self._num_heads,
                                     num_blocks=num_blocks, size_of_cols=self._feature_dim)
        model.summary()
        return model

    def get_last_train_hist(self):
        return self._last_train_hist

    def save_tf_model(self, model_path):
        self._model.save(model_path)

    @classmethod
    def build_n_grams_train_data(cls, sequence):
        input_seq = sequence[:-1]
        target_seq = sequence[1:]
        return input_seq, target_seq

    def vectorize_gene_information(self, sample: TrainingRecord):
        train_samples = []
        for index, chromosome in enumerate(sample.get_raw_pheno_type()):
            entry = []
            linking_functions = sample.get_linking_function()
            for elem in chromosome:
                entry.append(elem + self._link_token[ModelWrapper.gene_t_name_c])
            x_sequences_temp = [elem for sublist in entry for elem in sublist]
            entry = [self._chromosome_positions[index]] + linking_functions \
                    + self._link_token[ModelWrapper.link_t_name_c] + x_sequences_temp
            train_samples += entry
        units = [sample.get_unit_map().get(sym.replace('_', ''), np.zeros(7)) for sym in train_samples[:-1]]
        return self._char2int_(train_samples), units

    def learnDecay(self, epoch):
        if epoch < self.rampup_epochs:
            return (self.max_lr - self.start_lr) / self.rampup_epochs * epoch + self.start_lr
        elif epoch < self.rampup_epochs + self.sustain_epochs:
            return self.max_lr
        else:
            return (self.max_lr - self.min_lr) * self.exp_decay ** (
                    epoch - self.rampup_epochs - self.sustain_epochs) + self.min_lr

    def data_generator(self, replay_memory, batch_size, max_y_size):
        for i in range(0, len(replay_memory), batch_size):
            batch = replay_memory[i:i + batch_size]
            x_train_data = []
            y_train_data = []
            z_sample_weights = []
            embed_meta_info = []
            x_unit_pad = []
            for sample in batch:
                error = sample.get_error_information()
                emb = sample.get_meta_information()
                train_entries, unit_vecs = self.vectorize_gene_information(sample)
                x, y = ModelWrapper.build_n_grams_train_data(train_entries)

                # Use TensorFlow operations for padding
                x_padded = tf.pad(x, [[0, self._input_len - tf.shape(x)[0]]], constant_values=self._char2int_('0'))
                y_padded = tf.pad(y, [[0, self._input_len - tf.shape(y)[0]]], constant_values=self._char2int_('0'))

                x_unit_padded = tf.pad(unit_vecs, [[0, self._input_len - tf.shape(x)[0]],
                                                   [0, 0]], constant_values=0)
                x_train_data.append(x_padded)
                y_train_data.append(y_padded)
                x_unit_pad.append(x_unit_padded)
                z_sample_weights.append(tf.cast(tf.math.not_equal(y_padded, 0), tf.float32) * error)

                # Padding for embed_meta_info
                diff = max_y_size - tf.shape(emb)[0]
                padded_embed = tf.pad(emb, [[0, diff], [0, 0]], constant_values=0.0)
                embed_meta_info.append(padded_embed)

            # Shape(batch_size, self.input_len)
            x_train_arr = tf.stack(x_train_data)
            # Shape(batch_size, self.input_len)
            y_train_arr = tf.stack(y_train_data)
            # Shape(batch_size, self.embed_size)
            z_sample_weights = tf.stack(z_sample_weights)
            # Shape(batch_size, self.input_len, 7)
            embed_meta_info = tf.stack(embed_meta_info)
            x_unit_pad_arr = tf.stack(x_unit_pad)
            input_data = {
                self._model.input_names[0]: x_train_arr,
                self._model.input_names[1]: embed_meta_info,
                self._model.input_names[2]: x_unit_pad_arr
            }

            yield input_data, y_train_arr, z_sample_weights

    def _create_set(self, replay_memory, batch_size, max_y_size):
        return tf.data.Dataset.from_generator(
            lambda: self.data_generator(replay_memory, batch_size, max_y_size),
            output_signature=(
                {
                    self._model.input_names[0]: tf.TensorSpec(shape=(None, self._input_len), dtype=tf.int8),
                    self._model.input_names[1]: tf.TensorSpec(shape=(None, max_y_size, self._feature_dim),
                                                              dtype=tf.float32),
                    self._model.input_names[2]: tf.TensorSpec(shape=(None, self._input_len, 7),
                                                              dtype=tf.int8)
                },
                tf.TensorSpec(shape=(None, self._input_len), dtype=tf.int8),
                tf.TensorSpec(shape=(None, self._input_len), dtype=tf.float32)
            )
        )

    def set_learning_rate_params(self, **kwargs):
        self.__dict__.update(kwargs)
        print(f"Params: Start-rate={self.start_lr} Min-rate={self.min_lr} Max-rate={self.max_lr}")

    def train(self, replay_memory, **kwargs):
        np.random.shuffle(replay_memory)
        self.set_learning_rate_params(kwargs=kwargs)
        checkpoint_filepath = kwargs.get("check_point_path", "checkpoint.save")
        max_y_size = kwargs.get('max_y_size', 0)
        frequency = kwargs.get("frequency", 10)
        epochs = kwargs.get('epochs', 500)
        batch_size = min(kwargs.get("batch_size", 32), len(replay_memory))
        print(batch_size)
        test_size = int(len(replay_memory) * (1 - kwargs.get('test_size', 0.02)))
        train_set = self._create_set(replay_memory[:test_size], batch_size, max_y_size)
        train_set = train_set.repeat().prefetch(tf.data.AUTOTUNE)
        test_set = self._create_set(replay_memory[test_size:], batch_size, max_y_size)

        steps_per_epoch = len(replay_memory) // batch_size
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_weights_only=False,
            monitor='loss',
            save_freq=steps_per_epoch * frequency,
            mode='min',
            save_best_only=True,
            verbose=1
        )

        lr_callback = tf.keras.callbacks.LearningRateScheduler(lambda epoch: self.learnDecay(epoch), verbose=True)
        self._last_train_hist = self._model.fit(
            train_set,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            validation_data=test_set,
            validation_freq=kwargs.get('val_freq', 10000),
            callbacks=[model_checkpoint_callback, lr_callback],
            verbose=1, use_multiprocessing=True
        )
        del train_set
        return self._last_train_hist

    def sample_gene_seq(self, **kwargs):
        start_prompt = kwargs['state']
        kwargs.pop('state')
        if kwargs.get('link_sampling', False):
            start_prompt = [self._chromosome_positions[0]]
            ret_val, _, _ = self._sample_gene_seq(start_prompt, **kwargs)
            return [ret_val]
        else:
            linker = start_prompt[0] + self._link_token[ModelWrapper.link_t_name_c]
            genes = [elem + self._link_token[ModelWrapper.gene_t_name_c] for elem in start_prompt[1]]
            if len(genes) > 0:
                genes = genes[0]
            start_prompt = [self._chromosome_positions[0]] + linker + genes
            result = []
            uct = []
            for start_position in range(self._exp_count):
                ret_val, state, uncertainty = self._sample_gene_seq(start_prompt, **kwargs)
                result.append(ret_val)
                uct += uncertainty
            print(f"Perplex.: {np.exp(np.mean(np.array(uct)))}")
            return result

    @classmethod
    def sample_from(cls, logits, top_k):
        logits, indices = tf.math.top_k(logits, k=top_k, sorted=True)
        indices = np.asarray(indices).astype("int32")
        preds = keras.activations.softmax(tf.expand_dims(logits, 0))[0]
        preds = np.asarray(preds).astype("float32")
        num = np.random.choice(indices, p=preds)
        return num

    def _sample_gene_seq(self, state, **kwargs):
        """
        Steps:  -Create a prompt similar to the training data
                -Sample genes
                -Sample the linking symbols for the genes
        """
        embed = kwargs.get("embedding", tf.zeros((1, 1, self._feature_dim)))
        gene_count = kwargs.get('gene_count', 1)
        k_sampling = kwargs.get('k_sampling', self._number_k_sampling)
        temperature = kwargs.get('temperature', 1)
        prior_mask = self._prior_mask
        mask_syms = kwargs.get('mask_syms', [])

        perplex = []
        if kwargs.get('link_sampling', False):
            linked_symbols, state = self.sample(self._model, gene_count - 1, state,
                                                self._connector_prior_mask,
                                                embed,
                                                top_k=max(1, int(len(self._arity_syms_dict))),
                                                max_len=int(self._input_len), link_sampling=True,
                                                temperature=temperature,
                                                perplex=perplex)
            state += self._link_token[ModelWrapper.link_t_name_c]
            return [[], linked_symbols], state, perplex
        else:
            linked_symbols = []
            ret_genes, state = self.sample(self._model, gene_count, state, prior_mask,
                                           embed,
                                           top_k=k_sampling,
                                           max_len=int(self._input_len), temperature=temperature,
                                           perplex=perplex)

            list_chunked = self.pad_to_full_len(ret_genes, mask_syms)
            return [list_chunked, linked_symbols], state, perplex

    def pad_to_full_len(self, chunk, mask_syms=None):
        pad_array = np.where(np.isfinite(self._pad_array))
        pad_array = [self._vocab[i - 1] for i in pad_array[0]]
        if len(mask_syms) > 0:
            pad_array = list(set([elem for elem in pad_array if elem in mask_syms]))
        delim = self._link_token[ModelWrapper.gene_t_name_c][0]
        temp_chunk = [list(y) for x, y in itertools.groupby(chunk, lambda z: z == delim) if not x]
        target_gene_len = self._gene_size
        for elem in temp_chunk:
            if len(elem) != target_gene_len:
                [elem.append(pad_array[np.random.randint(len(pad_array))])
                 for _ in range(len(elem), target_gene_len)]
        return temp_chunk

    def sample_from_beam(self, logits):
        decoded_list, log_probabilities = tf.nn.ctc_beam_search_decoder(inputs=logits,
                                                                        sequence_length=tf.constant(
                                                                            np.ones(1),
                                                                            dtype=tf.int32),
                                                                        top_paths=self._beam_paths,
                                                                        beam_width=self._beam_width)
        token = [elem.values[0] for elem, prop in zip(decoded_list, log_probabilities[0].numpy()) if
                 len(elem.indices) > 0]

        props = [prop for elem, prop in zip(decoded_list, log_probabilities[0].numpy()) if len(elem.indices) > 0]
        try:
            return (token[0] if len(token) == 1 else token[ModelWrapper.sample_from(props, len(token))]).numpy()
        except Exception as e:
            return 0

    # retrieve a token list

    def sample(self, model, steps, start_tokens_, mask, embedding, top_k=10, max_len=9, idx=False,
               link_sampling=False, temperature=1, perplex=[]):
        temp_steps = 0
        start_tokens = [i for i in self._char2int_(start_tokens_).numpy()]
        tokens_generated = []
        arity_count = 0
        mask_index = 0
        while temp_steps < steps:
            pad_len = max_len - len(start_tokens)
            sample_index = len(start_tokens) - 1
            if pad_len <= 0:
                x = start_tokens[-self._input_len:]
                sample_index = max_len - 1
            elif pad_len > 0:
                x = start_tokens + [0] * pad_len
            else:
                x = start_tokens
            unit_syms = np.array([self._physical_units.get(sym, np.zeros(7)) for sym in x])
            x = tf.expand_dims(x, axis=0)
            if len(self._model.input_names) == 2:
                y, _ = model(
                    {self._model.input_names[0]: x,
                     self._model.input_names[1]: tf.expand_dims(embedding, axis=0)
                     },
                    False)
            else:
                y, _ = model(
                    {self._model.input_names[0]: x,
                     self._model.input_names[1]: tf.expand_dims(embedding, axis=0),
                     self._model.input_names[2]: tf.expand_dims(unit_syms, axis=0)
                     },
                    False)

            # Add the custom probability mask
            y = y / temperature
            y_target = tf.Variable(y[0].numpy(), dtype=tf.float32)

            y_target[sample_index].assign(y_target[sample_index] + mask[mask_index % len(mask)])
            mask_index += 1
            y_target_expand = tf.expand_dims(y_target, axis=1)
            uc = tf.expand_dims(y[0][sample_index].numpy(), 0)[0]
            perplex.append(uc)
            sample_token = 0
            if self._decoding_method == memco.BEAM_SEARCH and \
                    sample_index < (self._input_len - self._beam_width):
                sample_token = self.sample_from_beam(y_target_expand[sample_index:, :, :])

            if sample_token == 0:
                p_pred = smp.sample_top_p(y_target, self._top_p_value)[sample_index]
                sample_token = self.sample_from(p_pred, top_k)
            temp = self._int2char_(sample_token).numpy().decode("utf-8")

            if arity_count == 0 and temp in self._arity_syms_dict.keys():
                arity_count += self._arity_syms_dict[temp]
            elif temp in temp in self._arity_syms_dict.keys():
                arity_count += self._arity_syms_dict[temp]
                arity_count -= 1
            else:
                arity_count -= 1

            tokens_generated.append(sample_token)
            start_tokens.append(sample_token)

            if arity_count < 0:
                mask_index = 0
                start_tokens += [self._char2int_(self._link_token[ModelWrapper.gene_t_name_c][0]).numpy()]
                tokens_generated += [self._char2int_(self._link_token[ModelWrapper.gene_t_name_c][0]).numpy()]

            num_tokens_generated = len(tokens_generated)

            if num_tokens_generated > 0 and tokens_generated[-1] == self._char2int_(
                    self._link_token[ModelWrapper.gene_t_name_c][0]).numpy() \
                    and not link_sampling:
                temp_steps += 1
                arity_count = 0
            elif link_sampling:
                temp_steps += 1

        state = list(map(lambda i: self._int2char_(i).numpy().decode("utf-8"), start_tokens))
        txt = list(map(lambda i: self._int2char_(i).numpy().decode("utf-8"), tokens_generated))

        if not idx:
            return txt, state
        else:
            return start_tokens

    def update_params(self, **kwargs):
        self.__dict__.update(kwargs)

    def get_case_inference_vec(self):
        return self._case_inference_vec
