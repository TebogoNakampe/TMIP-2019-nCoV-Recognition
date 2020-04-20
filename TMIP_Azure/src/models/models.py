from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input, MaxPool2D, Conv2D, Flatten, LeakyReLU, BatchNormalization, concatenate
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.initializers import Constant


def dcnn_resnet(model_config, input_shape, metrics, n_classes=2, output_bias=None):
    '''
    Defines a deep convolutional neural network model for multiclass X-ray classification.
    :param model_config: A dictionary of parameters associated with the model architecture
    :param input_shape: The shape of the model input
    :param metrics: Metrics to track model's performance
    :return: a Keras Sequential model object with the architecture defined in this method
    '''

    # Set hyperparameters
    nodes_dense0 = model_config['NODES_DENSE0']
    lr = model_config['LR']
    dropout = model_config['DROPOUT']
    l2_lambda = model_config['L2_LAMBDA']
    if model_config['OPTIMIZER'] == 'adam':
        optimizer = Adam(learning_rate=lr)
    elif model_config['OPTIMIZER'] == 'sgd':
        optimizer = SGD(learning_rate=lr)
    else:
        optimizer = Adam(learning_rate=lr)  # For now, Adam is default option
    init_filters = model_config['INIT_FILTERS']
    filter_exp_base = model_config['FILTER_EXP_BASE']
    conv_blocks = model_config['CONV_BLOCKS']
    kernel_size = eval(model_config['KERNEL_SIZE'])
    max_pool_size = eval(model_config['MAXPOOL_SIZE'])
    strides = eval(model_config['STRIDES'])

    # Set output bias
    if output_bias is not None:
        output_bias = Constant(output_bias)
    print("MODEL CONFIG: ", model_config)

    # Input layer
    X_input = Input(input_shape)
    X = X_input

    # Add convolutional (residual) blocks
    for i in range(conv_blocks):
        X_res = X
        X = Conv2D(init_filters * (filter_exp_base ** i), kernel_size, strides=strides, padding='same',
                         kernel_initializer='he_uniform', activity_regularizer=l2(l2_lambda))(X)
        X = BatchNormalization()(X)
        X = LeakyReLU()(X)
        X = Conv2D(init_filters * (filter_exp_base ** i), kernel_size, strides=strides, padding='same',
                         kernel_initializer='he_uniform', activity_regularizer=l2(l2_lambda))(X)
        X = concatenate([X, X_res])
        X = BatchNormalization()(X)
        X = LeakyReLU()(X)
        X = MaxPool2D(max_pool_size, padding='same')(X)

    # Add fully connected layers
    X = Flatten()(X)
    X = Dropout(dropout)(X)
    X = Dense(nodes_dense0, kernel_initializer='he_uniform', activity_regularizer=l2(l2_lambda))(X)
    X = LeakyReLU()(X)
    Y = Dense(n_classes, activation='softmax', bias_initializer=output_bias, name='output')(X)

    # Set model loss function, optimizer, metrics.
    model = Model(inputs=X_input, outputs=Y)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=metrics)
    model.summary()
    return model