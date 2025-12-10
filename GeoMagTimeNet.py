# -*- coding: utf-8 -*-
# @Time : 2025/3/21 17:25
# @Author : Mengyu Wang
# @Email : 1179763088@qq.com
# @File : CNN_Transformer.py

import tensorflow as tf
from tensorflow.keras import layers, models
from keras.layers import Input, concatenate
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["TF_FORCR_GPU_ALLOW_GROWTH"] = "true"

# -------------------------------------------------------
# Swin Transformer Block (1D version)
# -------------------------------------------------------
class SwinTransformerBlock(layers.Layer):
    def __init__(self, dim, num_heads, **kwargs):
        super(SwinTransformerBlock, self).__init__(**kwargs)
        self.dim = dim
        self.num_heads = num_heads
        self.norm = layers.LayerNormalization()
        self.attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=dim)
        self.conv = layers.Conv1D(dim, 3, activation='gelu', padding='same')

    def call(self, inputs, training=False):
        x = self.norm(inputs)
        x = self.attn(x, x)
        x = self.conv(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "dim": self.dim,
            "num_heads": self.num_heads
        })
        return config


# -------------------------------------------------------
# Adaptive Fusion Block (mixing CNN + Swin features)
# -------------------------------------------------------
class AdaptiveFusionBlock(layers.Layer):
    def __init__(self, dim, **kwargs):
        super(AdaptiveFusionBlock, self).__init__(**kwargs)
        self.swin_transformer = SwinTransformerBlock(dim, num_heads=4)
        self.conv = layers.Conv1D(dim, kernel_size=3, padding="same", activation="relu")
        self.concat = layers.Concatenate()
        self.inputs_proj = layers.Conv1D(filters=dim * 2, kernel_size=1, padding='same')

    def call(self, inputs, training=False):
        swin = self.swin_transformer(inputs, training=training)
        conv = self.conv(inputs)
        concat = self.concat([swin, conv])
        inputs_proj = self.inputs_proj(inputs)
        return layers.Add()([inputs_proj, concat])

    def get_config(self):
        config = super().get_config()
        config.update({
            "dim": self.swin_transformer.dim
        })
        return config


# -------------------------------------------------------
# CATM: Cross-scale Attention Transition Module
# -------------------------------------------------------
class CATM(layers.Layer):
    def __init__(self, dim, **kwargs):
        super(CATM, self).__init__(**kwargs)
        self.dim = dim
        self.swin_transformer = SwinTransformerBlock(dim, num_heads=4)
        self.concat = layers.Concatenate()
        self.conv = layers.Conv1D(dim, kernel_size=1, activation="relu")

        # Weight mapping using Conv1D
        self.weight_conv_skip = layers.Conv1D(1, kernel_size=1, activation='sigmoid')
        self.weight_conv_decoder = layers.Conv1D(1, kernel_size=1, activation='sigmoid')

    def call(self, inputs, training=False):
        if not isinstance(inputs, (list, tuple)) or len(inputs) != 2:
            raise ValueError("CATM expects inputs = [x_skip, x_decoder]")

        x_skip, x_decoder = inputs
        swin = self.swin_transformer(x_decoder, training=training)

        weight_skip = self.weight_conv_skip(x_skip)
        weight_decoder = self.weight_conv_decoder(swin)

        weighted_skip = x_skip * weight_skip
        weighted_decoder = swin * weight_decoder

        concat = self.concat([weighted_skip, weighted_decoder])
        return self.conv(concat)

    def get_config(self):
        config = super().get_config()
        config.update({"dim": self.dim})
        return config


# -------------------------------------------------------
# Import Mamba, HAAM, FFT blocks
# -------------------------------------------------------
from mamba1 import Mamba



# -------------------------------------------------------
# Full Model
# -------------------------------------------------------
def build_model(input_shape=(1440, 3), output_classes=2):
    inputs = layers.Input(shape=input_shape)

    # ================= CNN Encoder =================
    x = layers.Conv1D(64, 3, padding='same', activation='relu')(inputs)
    x = layers.Conv1D(64, 3, padding='same', activation='relu')(x)
    f1 = layers.MaxPooling1D(2)(x)

    x = layers.Conv1D(128, 3, padding='same', activation='relu')(f1)
    x = layers.Conv1D(128, 3, padding='same', activation='relu')(x)
    f2 = layers.MaxPooling1D(2)(x)

    x = layers.Conv1D(256, 3, padding='same', activation='relu')(f2)
    x = layers.Conv1D(256, 3, padding='same', activation='relu')(x)
    f3 = layers.MaxPooling1D(2)(x)

    x = layers.Conv1D(512, 3, padding='same', activation='relu')(f3)
    x = layers.Conv1D(512, 3, padding='same', activation='relu')(x)
    f4 = layers.MaxPooling1D(2)(x)

    # ================= Mamba Transformer Encoder =================
    g1 = layers.Conv1D(256, 3, padding='same', activation='relu')(f1)
    g1 = Mamba(seq_len=1440, d_model=512, state_size=8)(g1)
    g11 = layers.MaxPooling1D(2)(g1)

    g2 = Mamba(seq_len=720, d_model=256, state_size=8)(g11)
    g22 = layers.MaxPooling1D(2)(g2)

    g3 = Mamba(seq_len=360, d_model=128, state_size=8)(g22)
    g33 = layers.MaxPooling1D(2)(g3)

    g4 = Mamba(seq_len=180, d_model=64, state_size=8)(g33)
    g44 = layers.MaxPooling1D(2)(g4)

    # ================= Cross-scale Feature Fusion =================
    m1 = CATM(64)([g1, f1])
    m2 = CATM(128)([g2, f2])
    m3 = CATM(256)([g3, f3])
    m4 = CATM(512)([g4, f4])

    # ================= Transformer Decoder =================
    u1 = AdaptiveFusionBlock(64)(m4)
    u1 = concatenate([u1, m4])
    u1 = layers.UpSampling1D(2)(u1)

    u2 = AdaptiveFusionBlock(256)(u1)
    u2 = concatenate([u2, m3])
    u2 = layers.UpSampling1D(2)(u2)

    u3 = AdaptiveFusionBlock(128)(u2)
    u3 = concatenate([u3, m2])
    u3 = layers.UpSampling1D(2)(u3)

    u4 = AdaptiveFusionBlock(32)(u3)
    u4 = concatenate([u4, m1])
    u4 = layers.UpSampling1D(2)(u4)

    outputs = layers.Conv1D(output_classes, kernel_size=1, activation='softmax')(u4)

    model = models.Model(inputs, outputs)
    return model


# -------------------------------------------------------
# Run Summary
# -------------------------------------------------------
if __name__ == "__main__":
    model = build_model()
    model.summary()
