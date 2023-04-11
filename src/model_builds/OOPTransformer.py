import sys, os, pickle
sys.path.append(os.path.realpath('../'))
# print(sys.path)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # INFO and WARNING messages are not printed

import tensorflow as tf

from typing import List, Any

from Transformer.Transformer import Transformer
from Transformer.CustomSchedule import CustomSchedule


class OOPTransformer:
    def __init__(self) -> None:
        self.model_name = 'OOP_Transformer'
        self.model = None
        self.history = None
        self.evaluation = None
        self.last_attn_scores = None
        self.file_name = f'../saved_models/{self.model_name}/'
        self.imgs_path = f'../saved_data/imgs/{self.model_name}/'
        self.histories_path = f'../saved_data/histories/{self.model_name}_history'


    def build(
            self,
            X_sample: Any,
            num_layers: int,
            d_model: int,
            dff: int,
            num_heads: int,
            dropout_rate: float,
            mlp_units: List[int],
            save_model: bool = True,
            verbose: bool = False
    ):
        self.model = Transformer(
            num_layers=num_layers,
            d_model=d_model,
            num_heads=num_heads,
            ff_dim=dff,
            mlp_units=mlp_units,
            input_space_size=6,
            target_space_size=2,
            training=True,
            dropout_rate=dropout_rate,
            pos_encoding=True
        )

        output = self.model(X_sample)
        attn_scores = self.model.encoder.enc_layers[-1].last_attn_scores
        if verbose:
            print(output.shape)
            print(attn_scores.shape)  # (batch, heads, target_seq, input_seq)
            print(self.model.summary())


        if save_model:
            self.model.save_weights(filepath=self.file_name)


    def compile(self):
        learning_rate = CustomSchedule()
        opt = tf.keras.optimizers.legacy.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                    epsilon=1e-9)

        self.model.compile(
            loss=tf.keras.losses.CategoricalCrossentropy(),
            optimizer=opt,
            metrics=[tf.keras.metrics.CategoricalAccuracy()]
        )


    def fit(
            self,
            X_train: Any,
            Y_train: Any,
            X_test: Any,
            Y_test: Any,
            callbacks: List[Any],
            epochs: int = 200,
            batch_size: int = 32,
            save_model: bool = True
    ):
        self.history = self.model.fit(
            x=X_train,
            y=Y_train,
            validation_split=0.1,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            validation_data = (X_test, Y_test),
            steps_per_epoch = len(X_train) // batch_size,
            validation_steps = len(X_test) // batch_size
        )

        self.last_attn_scores = self.model.encoder.enc_layers[-1].last_attn_scores

        if save_model:
            self.model.save_weights(filepath=self.file_name)

            with open(self.histories_path, 'wb') as file_pi:
                pickle.dump(self.history.history, file_pi)