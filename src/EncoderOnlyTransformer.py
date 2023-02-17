import os, sys, yaml, re
import tensorflow as tf
import tensorflow.keras as tfk
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras import regularizers
from YamlLoader import YamlLoader
from MultiHeadAttention import MultiHeadAttention


# https://keras.io/examples/timeseries/timeseries_transformer_classification/
class TransformerModel():
    def __init__(self, config: dict = None) -> None:
        self.path_to_config = 'src/config/model_config.yaml'
        if config == None:
            self.config = self.load_config()
        else:
            self.config = config
        self.model = None
        self.callbacks = None


    def load_config(self) -> dict:
        """ Load model config file using the Yaml loader class """
        yl = YamlLoader()
        conf = yl.load_yaml(path=self.path_to_config)
        if isinstance(conf, dict):
            return conf
        return None


    def get_model(self):
        return self.model


    def get_model_summary(self):
        return self.model.summary()


    def encoder(self, inputs):
        """ Build the transformer encoder """
        conf = self.config['encoder']

        # Normalization and Attention
        x = layers.LayerNormalization(epsilon=self.config['epsilon'])(inputs)
        # TODO: either use tf >= 2.4 where multi head attention layer is implemented or implement the layer myself
        # if using tensorflow >= 2.4:
        x = layers.MultiHeadAttention(
            key_dim=conf['head_size'],
            num_heads=conf['num_heads'],
            dropout=conf['dropout']
        )(x, x)
        # else:
        # x = MultiHeadAttention(
        #     h=conf['num_heads'],
        #     d_k=64, # ??
        #     d_v=64, # ??
        #     d_model=conf['head_size']
        # )(queries=??, keys=??, values=??)
        x = layers.Dropout(conf['dropout'])(x)
        res = x + inputs

        # Feed Forward Part
        x = layers.LayerNormalization(epsilon=self.config['epsilon'])(res)
        x = layers.Conv1D(filters=conf['ff_dim'], kernel_size=conf['kernel_size'], activation=conf['activation'])(x)
        x = layers.Dropout(conf['dropout'])(x)
        x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=conf['kernel_size'])(x)
        return x + res


    def build(self, input_shape, n_classes: int):
        """ Build the model """
        conf = self.config['build']
        inputs = tfk.Input(shape=input_shape) 
        x = inputs
        for _ in range(conf['num_transformer_blocks']):
            x = self.encoder(x)

        x = layers.GlobalAveragePooling1D(data_format='channels_first')(x)
        for dim in conf['mlp_units']:
            x = layers.Dense(dim, activation='relu')(x)
            x = layers.Dropout(conf['mlp_dropout'])(x)
        outputs = layers.Dense(n_classes, activation='softmax')(x)
        self.model = tfk.Model(inputs, outputs)


    def compile(self):
        conf = self.config['compile']
        self.model.compile(
            loss=conf['loss'],
            optimizer=tfk.optimizers.Adam(learning_rate=conf['lr']),
            metrics=conf['metrics']
        )


    def create_callbacks(self):
        conf = self.config['callbacks']
        self.callbacks = [tfk.callbacks.EarlyStopping(
            patience=conf['patience'],
            restore_best_weights=conf['best_w']
        )]


    def fit(self, x, y):
        conf = self.config['fit']
        self.model.fit(
            x,
            y,
            validation_split=conf['val_split'],
            epochs=conf['epochs'],
            batch_size=conf['batch_size'],
            callbacks=self.callbacks
        )


    def evaluate(self, x, y):
        self.model.evaluate(x, y, verbose=1)







if __name__ == '__main__':
    print( sys.version )
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # INFO and WARNING messages are not printed
    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    print( f"Found {len(gpus)} GPUs!" )
    for i in range( len( gpus ) ):
        try:
            tf.config.experimental.set_memory_growth(device=gpus[i], enable=True)
            tf.config.experimental.VirtualDeviceConfiguration( memory_limit = 1024*3 )
            print( f"\t{tf.config.experimental.get_device_details( device=gpus[i] )}" )
        except RuntimeError as e:
            print( '\n', e, '\n' )

    devices = tf.config.list_physical_devices()
    print( "Tensorflow sees the following devices:" )
    for dev in devices:
        print( f"\t{dev}" )
    
    MAIN_PATH = os.path.dirname(os.path.dirname(__file__))
    SRC_PATH = os.path.dirname(os.path.realpath(__file__))
