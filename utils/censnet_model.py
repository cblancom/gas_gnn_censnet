from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Layer,
    Input,
    Normalization,
    Dense,
    LeakyReLU,
    BatchNormalization,
    ReLU,
)
import tensorflow as tf
from spektral.layers import CensNetConv
import numpy as np


class BalanceLayer(Layer):
    def __init__(self, **kwargs):
        super(BalanceLayer, self).__init__(**kwargs)

    def call(self, inputs):
        # Ensure that inputs is a list of 11 tensors
        if len(inputs) != 4:
            raise ValueError(
                "CustomLayer expects 4 input tensors, got {}".format(len(inputs))
            )

        X, preds_n, preds_e, IR = inputs
        balance = (IR @ preds_e)[:, :, 0] - X[:, :, 2] + preds_n[:, :, 0]
        return balance


class WeymouthLayer(Layer):
    def __init__(self, **kwargs):
        super(WeymouthLayer, self).__init__(**kwargs)

    def call(self, inputs):
        # Ensure that inputs is a list of 11 tensors
        if len(inputs) != 4:
            raise ValueError(
                "CustomLayer expects 4 input tensors, got {}".format(len(inputs))
            )

        yn, ye, Z, IR = inputs

        press = (
            tf.math.pow(Z[:, :, 0], 2)
            * tf.linalg.matmul(
                -1 * IR, tf.math.pow(yn * 100, 2), transpose_a=True, transpose_b=False
            )[:, :, 0]
        )

        flow = (Z[:, :, 0]) * (tf.math.pow(ye[:, :, 0], 2) * tf.math.sign(ye[:, :, 0]))
        return press - flow


class CensNetModel:
    def __init__(
        self,
        N_layers=5,
        N_channels=128,
        act_func=None,
        N_dense=2,
        seed=5,
        Node_size=63,
        Edge_size=62,
    ):
        self.N_layers = N_layers
        self.N_channels = N_channels
        self.act_func = act_func if act_func else LeakyReLU(alpha=0.2)
        self.N_dense = N_dense
        self.seed = seed
        self.Node_size = Node_size
        self.Edge_size = Edge_size
        self.build_model()

    def build_model(self):
        N = Input(shape=(self.Node_size, 5), name="node_features")
        NL = Input(shape=(self.Node_size, self.Node_size), name="node_lap")
        EL = Input(shape=(self.Edge_size, self.Edge_size), name="edge_lap")
        I = Input(shape=(self.Node_size, self.Edge_size), name="incid")
        E = Input(shape=(self.Edge_size, 4), name="edge_feats")

        norm_node = Normalization()(N)
        norm_edge = Normalization()(E)

        norm_node = Dense(
            self.N_channels,
            activation=self.act_func,
            kernel_initializer=glorot_uniform(seed=self.seed),
            name="node_predense_0",
        )(norm_node)
        norm_edge = Dense(
            self.N_channels,
            activation=self.act_func,
            kernel_initializer=glorot_uniform(seed=self.seed),
            name="edge_predense_0",
        )(norm_edge)
        norm_node = Dense(
            self.N_channels,
            activation=self.act_func,
            kernel_initializer=glorot_uniform(seed=self.seed),
            name="node_predense_1",
        )(norm_node)
        norm_edge = Dense(
            self.N_channels,
            activation=self.act_func,
            kernel_initializer=glorot_uniform(seed=self.seed),
            name="edge_predense_1",
        )(norm_edge)

        inputs = [norm_node, [NL, EL, I], norm_edge]
        for i in range(1, self.N_layers + 1):
            out = CensNetConv(
                node_channels=self.N_channels,
                edge_channels=self.N_channels,
                activation=self.act_func,
                kernel_initializer=glorot_uniform(seed=self.seed),
                name=f"conv{i}",
            )(inputs)

            norm_node = BatchNormalization()(out[0])
            norm_edge = BatchNormalization()(out[1])
            inputs = [norm_node, [NL, EL, I], norm_edge]

        if self.N_dense <= 0:
            out = CensNetConv(
                node_channels=2,
                edge_channels=1,
                kernel_initializer=glorot_uniform(seed=self.seed),
                name="out",
            )(inputs)
            out_fw = out[0]
            out_p = out[0]
            out_2 = out[1]
            out_3 = BalanceLayer(name="balance")([N, out_fw, out_2, I])
            out_4 = WeymouthLayer(name="weymouth")([out_p, out_2, E, I])

        elif self.N_dense == 1:
            out = CensNetConv(
                node_channels=self.N_channels,
                edge_channels=self.N_channels,
                activation=self.act_func,
                kernel_initializer=glorot_uniform(seed=self.seed),
                name="out",
            )(inputs)

            out_fw = Dense(
                1,
                activation=None,
                name="fw",
                kernel_initializer=glorot_uniform(seed=self.seed),
            )(out[0])
            out_p = Dense(
                1,
                activation=None,
                name="p",
                kernel_initializer=glorot_uniform(seed=self.seed),
            )(out[0])

            out_2 = Dense(
                1,
                activation=None,
                name="edge",
                kernel_initializer=glorot_uniform(seed=self.seed),
            )(out[1])
            out_3 = BalanceLayer(name="balance")([N, out_fw, out_2, I])
            out_4 = WeymouthLayer(name="weymouth")([out_p, out_2, E, I])

        else:
            out = CensNetConv(
                node_channels=self.N_channels,
                edge_channels=self.N_channels,
                activation=self.act_func,
                name="out",
            )(inputs)
            out_1 = Dense(
                self.N_channels,
                activation=self.act_func,
                kernel_initializer=glorot_uniform(seed=self.seed),
                name="node_postdense_0",
            )(out[0])
            out_2 = Dense(
                self.N_channels,
                activation=self.act_func,
                kernel_initializer=glorot_uniform(seed=self.seed),
                name="edge_postdense_0",
            )(out[1])

            for i in range(self.N_dense - 2):
                out_1 = Dense(
                    self.N_channels,
                    activation=None,
                    kernel_initializer=glorot_uniform(seed=self.seed),
                    name=f"node_postdense_{i + 1}",
                )(out_1)
                out_2 = Dense(
                    self.N_channels,
                    activation=None,
                    kernel_initializer=glorot_uniform(seed=self.seed),
                    name=f"edge_postdense_{i + 1}",
                )(out_2)

            out_fw = Dense(
                1,
                activation=None,
                name="fw",
                kernel_initializer=glorot_uniform(seed=self.seed),
            )(out_1)
            out_p = Dense(
                1,
                activation=None,
                name="p",
                kernel_initializer=glorot_uniform(seed=self.seed),
            )(out_1)
            out_2 = Dense(
                1,
                activation=None,
                name="edge",
                kernel_initializer=glorot_uniform(seed=self.seed),
            )(out_2)
            out_3 = BalanceLayer(name="balance")([N, out_fw, out_2, I])
            out_4 = WeymouthLayer(name="weymouth")([out_p, out_2, E, I])

        model = Model(
            inputs=(N, NL, EL, I, E),
            outputs=[out_fw, out_p, out_2, out_3, out_4],
            name="model",
        )

        return model


def eval_nrmse(y_true, y_pred):
    true_var = np.nanvar(y_true, axis=0)
    series_length = y_pred.shape[0]
    nrmse_vals = np.sqrt(
        (np.sum((y_pred - y_true) ** 2, axis=0) / float(series_length)) / (true_var)
    )
    return np.nanmean(nrmse_vals)
