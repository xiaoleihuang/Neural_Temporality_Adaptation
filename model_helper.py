from keras.layers import Input, Embedding
from keras.layers import Bidirectional, GRU, Layer, LSTM
from keras.layers import Dense, Dropout, Lambda
from keras.layers import Conv1D, MaxPool1D, Flatten
from keras.layers import BatchNormalization
import keras
from keras.models import Model
import keras.backend as K
import tensorflow as tf
from keras import optimizers


def mask_layer(inputs):
    """
    This function will be applied to Lambda, only select the domain features
    (x_d, 0, 0)
    :param inputs:
    :return:
    """
    return keras.backend.squeeze(keras.backend.dot(inputs[-1], inputs[:-1]))


def create_rnn(wt_generator, config):
    """

    :param wt_generator: a tuple generator of weights for embedding: key, weight_matrix
    :type wt_generator: tuple
    :param config: configurations
    :type config: dict
    :return:
    """
    # define the inputs
    inputs = []
    # define the embeds
    embeds = []
    # define the bi-directional lstm
    bilstms = []

    # loop through each domain
    for idx, key, wt_matrix in wt_generator:
        inputs.append(Input(shape=(int(config['seq_max_len']),), dtype='int32', name='input_'+str(key)))

        embeds.append(
            Embedding(
                wt_matrix.shape[0],
                wt_matrix.shape[1],
                weights=[wt_matrix],
                input_length=int(config['seq_max_len']),
                trainable=True,
                name='embed_'+str(key)
            )(inputs[idx])
        )
        bilstms.append(
            Bidirectional(
                GRU(
                    int(config['rnn_size']),
                    kernel_initializer="glorot_uniform",
                    kernel_regularizer=keras.regularizers.l1_l2(float(config['l1_rate']), float(config['l2_rate'])),
                    recurrent_activation=config['rnn_ac']
                ),
                name='lstm_' + str(key)
            )(embeds[idx])
        )

    merge_vec = keras.layers.concatenate(bilstms, axis=-1)
    merge_dp = Dropout(float(config['dp_rate']), name='merge_dp')(merge_vec)
    merge_dense = Dense(
        int(config['dense_size']),
        activation=config['dense_ac'],
        kernel_regularizer=keras.regularizers.l1_l2(float(config['l1_rate']), float(config['l2_rate'])),
        name='merge_dense'
    )(merge_dp)
    predicts = Dense(int(config['num_cl']), activation='sigmoid', name='final_dp')(merge_dense)

    rnn_model = Model(inputs=inputs, outputs=predicts)
    rnn_model.compile(
        loss=config['loss'], optimizer= config['opt'],
        metrics=['accuracy']
    )
    print(rnn_model.summary())

    return rnn_model


def create_cnn(wt_generator, config):
    """

    :param wt_generator:
    :param config:
    :return:
    """
    # define the inputs
    inputs = []
    # define the embeddings
    embeds = []
    # normalize
    normals= [] # use the BatchNormalization ????
    # define the convolution layers
    convs = []
    # define pooling layers
    pools = []

    # loop through each domain
    for idx, key, wt_matrix in wt_generator:
        inputs.append(Input(shape=(int(config['seq_max_len']),), dtype='int32', name='input_' + str(key)))

        embeds.append(
            Embedding(
                wt_matrix.shape[0],
                wt_matrix.shape[1],
                weights=[wt_matrix],
                input_length=int(config['seq_max_len']),
                trainable=True,
                name='embed_' + str(key)
            )(inputs[idx])
        )

        convs.append(Conv1D(
            filters=config['filters'],
            kernel_size=config['kernel_size'],
            padding=config['padding'],
            strides=config['strides'],
            kernel_regularizer=keras.regularizers.l1_l2(float(config['l1_rate']), float(config['l2_rate'])),
        )(embeds[idx]))
        pools.append(
            MaxPool1D(config['pool_size'], padding=config['padding'])(convs[idx])
        )

    merge_vec = keras.layers.concatenate(pools, axis=-1)
    merge_dp = Dropout(config['dp_rate'], name='mergedp')(merge_vec)

    merge_dense = Dense(int(
        config['dense_size']),
        activation=config['dense_ac'],
        kernel_regularizer=keras.regularizers.l1_l2(
            float(config['l1_rate']), float(config['l2_rate'])),
        name='merge_dense')(merge_dp)

    flatten = Flatten(name='flatten')(merge_dense)

    predicts = Dense(int(config['num_cl']), activation='sigmoid', name='final_dp')(flatten)

    cnn_model = Model(inputs=inputs, outputs=predicts)
    cnn_model.compile(
        loss=config['loss'], optimizer=config['opt'],
        metrics=['accuracy']
    )
    print(cnn_model.summary())

    return cnn_model


def create_cnn_rnn(wt_generator, config):
    # define the inputs
    inputs = []
    # define the embeddings
    embeds = []
    # normalize
    normals = []  # use the BatchNormalization ????
    # define the convolution layers
    convs = []
    # define pooling layers
    pools = []
    # RNN for the pooling layer
    bilstms = []

    # loop through each domain
    for idx, key, wt_matrix in wt_generator:
        inputs.append(Input(shape=(int(config['seq_max_len']),), dtype='int32', name='input_' + str(key)))

        embeds.append(
            Embedding(
                wt_matrix.shape[0],
                wt_matrix.shape[1],
                weights=[wt_matrix],
                input_length=int(config['seq_max_len']),
                trainable=True,
                name='embed_' + str(key)
            )(inputs[idx])
        )

        convs.append(Conv1D(
            filters=config['filters'],
            kernel_size=config['kernel_size'],
            padding=config['padding'],
            strides=config['strides'],
            kernel_regularizer=keras.regularizers.l1_l2(float(config['l1_rate']), float(config['l2_rate'])),
        )(embeds[idx]))
        pools.append(
            MaxPool1D(config['pool_size'], padding=config['padding'])(convs[idx])
        )
        bilstms.append(
            Bidirectional(
                GRU(
                    int(config['rnn_size']),
                    kernel_initializer="glorot_uniform",
                    kernel_regularizer=keras.regularizers.l1_l2(float(config['l1_rate']), float(config['l2_rate'])),
                    recurrent_activation=config['rnn_ac']
                ),
                name='bigru_' + str(key)
            )(pools[idx])
        )

    merge_vec = keras.layers.concatenate(bilstms, axis=-1)
    # merge_lstm = Bidirectional(
    #     GRU(
    #         int(config['rnn_size']),
    #         kernel_initializer="glorot_uniform",
    #         kernel_regularizer=keras.regularizers.l1_l2(float(config['l1_rate']), float(config['l2_rate'])),
    #         recurrent_activation=config['rnn_ac']
    #     )
    # )(merge_vec)
    merge_dp = Dropout(config['dp_rate'], name='mergedp')(merge_vec)

    merge_dense = Dense(int(
        config['dense_size']),
        activation=config['dense_ac'],
        kernel_regularizer=keras.regularizers.l1_l2(
            float(config['l1_rate']), float(config['l2_rate'])),
        name='merge_dense')(merge_dp)

    # flatten = Flatten(name='flatten')(merge_dense)

    predicts = Dense(int(config['num_cl']), activation='sigmoid', name='final_dp')(merge_dense)

    cnn_gru_model = Model(inputs=inputs, outputs=predicts)
    cnn_gru_model.compile(
        loss=config['loss'], optimizer=config['opt'],
        metrics=['accuracy']
    )
    print(cnn_gru_model.summary())

    return cnn_gru_model


def create_hawkes(wt_generator, config):
    # define the inputs
    inputs = []
    # define the embeds
    embeds = []
    # define the bi-directional lstm
    bilstms = []
    encoder_states = []

    # loop through each domain
    for idx, key, wt_matrix in wt_generator:
        inputs.append(
            Input(
                shape=(int(config['seq_max_len']),), 
                dtype='int32', name='input_' + str(key),
            )
        )

        embeds.append(
            Embedding(
                wt_matrix.shape[0],
                wt_matrix.shape[1],
                weights=[wt_matrix],
                input_length=int(config['seq_max_len']),
                trainable=True,
                name='embed_' + str(key)
            )(inputs[idx])
        )

        # if it is the first one, 
        # the encoder will not be initialized by previous hidden states
        # otherwise, the encoder will be initialized by previous states   

        if key != 'general':
            encoder = LSTM(
                    int(config['rnn_size']), return_state=True, 
                    return_sequences=True,
                    kernel_initializer="glorot_uniform",
                    kernel_regularizer=keras.regularizers.l1_l2(
                        float(config['l1_rate']), 
                        float(config['l2_rate'])
                    ),
                    dropout=float(config['dp_rate']),
                    recurrent_activation=config['rnn_ac'],
                    name='lstm_' + str(key)
                )

            b_encoder = LSTM(
                    int(config['rnn_size']), return_state=True, 
                    return_sequences=True,
                    kernel_initializer="glorot_uniform",
                    kernel_regularizer=keras.regularizers.l1_l2(
                        float(config['l1_rate']), 
                        float(config['l2_rate'])
                    ),
                    dropout=float(config['dp_rate']),
                    go_backwards=True,
                    recurrent_activation=config['rnn_ac'],
                    name='blstm_' + str(key)
                )

            # if the current key is general domain, only the outputs
            if idx == 0:
                opts, f_h, f_c = encoder(embeds[idx])
                b_opts, b_h, b_c = b_encoder(embeds[idx])
            else:
                # the outputs from bidirectional rnn, 
                # include outputs, forward hidden states and cell memory;
                # backward hidden states and cell memory
                opts, f_h, f_c = encoder(embeds[idx], initial_state=encoder_states)
                b_opts, b_h, b_c = b_encoder(embeds[idx], initial_state=b_encoder_states)

            encoder_states = [f_h, f_c]
            b_encoder_states = [b_h, b_c]
        else:
            encoder = LSTM(
                    int(config['rnn_size']),
                    kernel_initializer="glorot_uniform",
                    kernel_regularizer=keras.regularizers.l1_l2(
                        float(config['l1_rate']), 
                        float(config['l2_rate'])
                    ),
                    dropout=float(config['dp_rate']),
                    recurrent_activation=config['rnn_ac'],
                    name='lstm_' + str(key)
                )(embeds[idx], initial_state=encoder_states)

            b_encoder = LSTM(
                    int(config['rnn_size']),
                    kernel_initializer="glorot_uniform",
                    kernel_regularizer=keras.regularizers.l1_l2(
                        float(config['l1_rate']), 
                        float(config['l2_rate'])
                    ),
                    dropout=float(config['dp_rate']),
                    go_backwards=True,
                    recurrent_activation=config['rnn_ac'],
                    name='blstm_' + str(key)
                )(embeds[idx], initial_state=b_encoder_states)

    # define the sentiment classification task
    dense_senti = Dense(
        int(config['dense_size']),
        activation='relu', # config['dense_ac'],
        kernel_regularizer=keras.regularizers.l1_l2(float(config['l1_rate']), float(config['l2_rate'])),
        name='dense_senti'
    )(keras.layers.concatenate([encoder, b_encoder]))

    opt = keras.optimizers.RMSprop(lr=0.0001)
#    opt = keras.optimizers.Adam(lr=0.0003)

    if config['pred_num'] < 3:
        pred_senti = Dense(1, activation='sigmoid', name='senti')(dense_senti)
        my_model = Model(inputs=inputs, outputs=pred_senti)
        my_model.compile(
            loss='binary_crossentropy',
            optimizer=opt,
            metrics=['accuracy']
        )
    else:
        pred_senti = Dense(config['pred_num'], activation='softmax', name='senti')(dense_senti)
        my_model = Model(inputs=inputs, outputs=pred_senti)
        my_model.compile(
            loss='categorical_crossentropy',
            optimizer=opt,
            metrics=['accuracy']
        )

    print(my_model.summary())
    return my_model
