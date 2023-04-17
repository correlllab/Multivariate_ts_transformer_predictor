import sys, os, pickle
sys.path.append(os.path.realpath('../'))
# print(sys.path)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # INFO and WARNING messages are not printed

import tensorflow as tf

from typing import List, Any

from Transformer.Transformer import Transformer
from Transformer.CustomSchedule import CustomSchedule


class OOPTransformer:
    def __init__(self, model_name: str = 'OOP_Transformer') -> None:
        self.model_name = model_name
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
            ff_dim: int,
            num_heads: int,
            head_size: int,
            dropout_rate: float,
            mlp_dropout: float,
            mlp_units: List[int],
            save_model: bool = True,
            verbose: bool = False
    ):
        self.model = Transformer(
            num_layers=num_layers,
            d_model=d_model,
            num_heads=num_heads,
            head_size=head_size,
            ff_dim=ff_dim,
            mlp_units=mlp_units,
            input_space_size=6,
            target_space_size=2,
            training=True,
            dropout_rate=dropout_rate,
            mlp_dropout=mlp_dropout,
            pos_encoding=True
        )

        output = self.model(X_sample)
        attn_scores = self.model.encoder.enc_layers[-1].last_attn_scores
        if verbose:
            print(output.shape)
            print(attn_scores.shape)  # (batch, heads, target_seq, input_seq)
            print(self.model.summary())


    def compile(self):
        # learning_rate = CustomSchedule()
        learning_rate = 1e-4
        opt = tf.keras.optimizers.legacy.Adam(learning_rate)
        opt = tf.keras.mixed_precision.LossScaleOptimizer(opt)
        loss_object = tf.keras.losses.CategoricalCrossentropy()

        self.model.compile(
            loss=loss_object,
            optimizer=opt,
            metrics=[tf.keras.metrics.CategoricalAccuracy()]
        )


    def fit(
            self,
            X_train: Any,
            Y_train: Any,
            X_test: Any,
            Y_test: Any,
            epochs: int = 200,
            batch_size: int = 64,
            save_model: bool = True
    ):
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                start_from_epoch=epochs*0.2
            )
        ]
        self.history = self.model.fit(
            x=X_train,
            y=Y_train,
            validation_split=0.1,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            validation_data=(X_test, Y_test),
            steps_per_epoch=len(X_train) // batch_size,
            validation_steps=len(X_test) // batch_size
        )

        self.last_attn_scores = self.model.encoder.enc_layers[-1].last_attn_scores

        if save_model:
            self.model.save_weights(filepath=self.file_name)

            with open(self.histories_path, 'wb') as file_pi:
                pickle.dump(self.history.history, file_pi)