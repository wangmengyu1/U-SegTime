# -*- coding: utf-8 -*-
# @时间 : 2025/4/1 11:23
# @作者 : 王梦雨
# @Email : 1179763088@qq.com
# @File : mamba.py
# @Project : 基建_code
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# class RMSNorm(layers.Layer):
#     def __init__(self, d_model, eps=1e-5):
#         super(RMSNorm, self).__init__()
#         self.eps = eps
#         self.weight = self.add_weight(shape=(d_model,), initializer="ones", trainable=True)
#
#     def call(self, x):
#         return x * tf.math.rsqrt(tf.reduce_mean(tf.square(x), axis=-1, keepdims=True) + self.eps) * self.weight

import tensorflow as tf
from tensorflow.keras import layers


class RMSNorm(layers.Layer):
    def __init__(self, eps=1e-5, **kwargs):
        super(RMSNorm, self).__init__(**kwargs)
        self.eps = eps
        self.weight = None  # 延迟初始化

    def build(self, input_shape):
        d_model = input_shape[-1]  # 获取输入的特征维度
        self.weight = self.add_weight(shape=(d_model,), initializer="ones", trainable=True)

    def call(self, x):
        return x * tf.math.rsqrt(tf.reduce_mean(tf.square(x), axis=-1, keepdims=True) + self.eps) * self.weight

    def get_config(self):
        config = super(RMSNorm, self).get_config()
        config.update({"eps": self.eps})
        return config


class S6(layers.Layer):
    def __init__(self, seq_len, d_model, state_size, **kwargs):
        super(S6, self).__init__(**kwargs)
        self.seq_len = seq_len
        self.d_model = d_model
        self.state_size = state_size

        # self.fc1 = layers.Dense(d_model)
        # self.fc2 = layers.Dense(state_size)
        # self.fc3 = layers.Dense(state_size)
        self.fc1 = layers.Conv1D(filters= d_model, kernel_size=3, padding="same")
        self.fc2 = layers.Conv1D(filters= state_size, kernel_size=3, padding="same")
        self.fc3 = layers.Conv1D(filters= state_size, kernel_size=3, padding="same")
        self.A = self.add_weight(shape=(d_model, state_size), initializer="glorot_uniform", trainable=True)

    def call(self, x):
        B = self.fc2(x)
        C = self.fc3(x)
        delta = tf.nn.softplus(self.fc1(x))
        dB = tf.einsum("bld,bln->bldn", delta, B)
        dA = tf.exp(tf.einsum("bld,dn->bldn", delta, self.A))
        h = tf.zeros_like(dB)
        h = tf.einsum('bldn,bldn->bldn', dA, h) + tf.expand_dims(x, -1) * dB
        y = tf.einsum('bln,bldn->bld', C, h)
        return y

    def get_config(self):
        config = super(S6, self).get_config()
        config.update({
            "seq_len": self.seq_len,
            "d_model": self.d_model,
            "state_size": self.state_size,
        })
        return config


class MambaBlock(layers.Layer):
    def __init__(self, seq_len, d_model, state_size, **kwargs):
        super(MambaBlock, self).__init__(**kwargs)
        self.norm = RMSNorm()
        # self.inp_proj = layers.Dense(2 * d_model)
        # self.out_proj = layers.Dense(d_model)
        self.inp_proj = layers.Conv1D(filters=2 * d_model, kernel_size=3, padding="same")
        self.out_proj = layers.Conv1D(filters=d_model, kernel_size=3, padding="same",)
        self.conv = layers.Conv1D(filters=2 * d_model, kernel_size=3, padding="same", activation="swish")
        # self.conv_linear = layers.Dense(2 * d_model)
        self.conv_linear = layers.Conv1D(filters=2 * d_model, kernel_size=3, padding="same")
        self.s6 = S6(seq_len, 2 * d_model, state_size)
        # self.residual = layers.Dense(2 * d_model, activation="swish")
        self.residual = layers.Conv1D(filters=2 * d_model, kernel_size=3, padding="same", activation="swish")

    def call(self, x):
        x = self.norm(x)
        print("x",x.shape)
        x_proj = self.inp_proj(x)
        print("x_proj",x_proj.shape)
        x_conv = self.conv(x_proj)
        print("x_conv",x_conv.shape)
        x_conv_out = self.conv_linear(x_conv)
        print("x_conv_out",x_conv_out.shape)
        x_ssm = self.s6(x_conv_out)
        print("x_ssm",x_ssm.shape)
        x_residual = self.residual(x)
        print("x_residual",x_residual.shape)
        x_combined = x_ssm * x_residual
        print("x_combined",x_combined.shape)
        return self.out_proj(x_combined)

    def get_config(self):
        config = super(MambaBlock, self).get_config()
        config.update({
            "seq_len": self.s6.seq_len,
            "d_model": self.s6.d_model,
            "state_size": self.s6.state_size,
        })
        return config
# class MambaBlock(layers.Layer):
#     def __init__(self, seq_len, d_model, state_size, **kwargs):
#         super(MambaBlock, self).__init__(**kwargs)
#         self.norm = RMSNorm()
#         self.inp_proj = layers.Conv1D(filters=d_model, kernel_size=3, padding="same")
#         self.out_proj = layers.Conv1D(filters=d_model, kernel_size=3, padding="same")
#         self.conv = layers.Conv1D(filters=d_model, kernel_size=3, padding="same", activation="swish")
#         self.conv_linear = layers.Conv1D(filters=d_model, kernel_size=3, padding="same")
#         self.s6 = S6(seq_len, d_model, state_size)
#         self.residual = layers.Conv1D(filters=d_model, kernel_size=3, padding="same", activation="swish")
#
#     def call(self, x_input):
#         x = self.norm(x_input)
#         print('1',x.shape)
#         # 分支 1：x -> Conv1D -> SSM
#         x_proj = self.inp_proj(x)
#         print('2',x_proj.shape)
#         x_ssm = self.s6(x_proj)
#         print('3',x_ssm.shape)
#
#         # 分支 2：z -> Activation -> Conv1D
#         z = self.residual(x)  # 这个 residual 实际就是图中的 Conv1D + Activation
#         print('4',z.shape)
#         # 相乘：x_ssm ⊙ z
#         y = x_ssm * z
#         print('5',y.shape)
#         # out_proj
#         y_out = self.out_proj(y)
#         print('6',y_out.shape)
#
#         # 残差连接
#         return x_input + y_out

    # def get_config(self):
    #     config = super(MambaBlock, self).get_config()
    #     config.update({
    #         "seq_len": self.s6.seq_len,
    #         "d_model": self.s6.d_model,
    #         "state_size": self.s6.state_size,
    #     })
    #     return config


class Mamba(layers.Layer):
    def __init__(self, seq_len, d_model, state_size, **kwargs):
        super(Mamba, self).__init__(**kwargs)
        self.mamba_block1 = MambaBlock(seq_len, d_model, state_size)
        self.mamba_block2 = MambaBlock(seq_len, d_model, state_size)
        self.mamba_block3 = MambaBlock(seq_len, d_model, state_size)
        self.conv1d = layers.Conv1D(filters=d_model, kernel_size=1)

    def call(self, x):
        x = self.mamba_block1(x)
        x = self.mamba_block2(x)
        x = self.mamba_block3(x)
        return self.conv1d(x)

    def get_config(self):
        config = super(Mamba, self).get_config()
        config.update({
            "seq_len": self.mamba_block1.s6.seq_len,
            "d_model": self.mamba_block1.s6.d_model,
            "state_size": self.mamba_block1.s6.state_size,
        })
        return config


if __name__ == "__main__":
    seq_len = 1440  # 设定序列长度
    d_model = 3   # 特征维度
    state_size = 16  # S6状态维度
    batch_size = 2  # 测试批次大小

    # 创建 Mamba 模型
    model = Mamba(seq_len, d_model, state_size)

    # 生成随机输入 (batch_size, seq_len, d_model)
    x_test = tf.random.normal((batch_size, seq_len, d_model))

    # 运行模型
    y_test = model(x_test)

    # 打印输入和输出的形状
    print(f"Input shape: {x_test.shape}")
    print(f"Output shape: {y_test.shape}")  # 期望输出形状应为 (batch_size, seq_len, d_model)
