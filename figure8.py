# -*- coding: utf-8 -*-
"""
This script evaluates multiple trained models on the test dataset,
calculates various performance metrics (e.g., accuracy, precision, recall, F1-score, AUC, mIoU, Dice),
and plots the ROC and Precision-Recall curves for each model.

The resulting ROC curves correspond to **Figure 8** in the manuscript.
"""

import os
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
import matplotlib as mpl
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import Loss
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, average_precision_score,
    precision_recall_curve, confusion_matrix
)
import tensorflow_addons as tfa
from CNN_Transformer import SwinTransformerBlock, AdaptiveFusionBlock, CATM
from mamba1 import Mamba
from u2net import REBNCONV

# Weighted categorical crossentropy loss
def weighted_categorical_crossentropy(weights):
    weights = tf.constant(weights, dtype=tf.float32)
    def loss(y_true, y_pred):
        y_pred /= tf.reduce_sum(y_pred, axis=-1, keepdims=True)
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
        loss = y_true * tf.math.log(y_pred) * weights
        return -tf.reduce_sum(loss, -1)
    return loss

custom_loss = weighted_categorical_crossentropy([0.5, 2])

# GPU configuration
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

# Paths
test_geo_path = '../data/test_data/'  # Update as needed
test_label_path = '../data/test_label/'  # Update as needed

# Load main model with custom layers
main_model_path = os.path.abspath('/external_data/duibi/compare_jijian/u-segtime.h5')
if not os.path.exists(main_model_path):
    raise FileNotFoundError(f"Model file not found at {main_model_path}")

model = load_model(main_model_path, compile=False, custom_objects={
    'weighted_categorical_crossentropy': custom_loss,
    'CATM': CATM,
    'SwinTransformerBlock': SwinTransformerBlock,
    'AdaptiveFusionBlock': AdaptiveFusionBlock,
    'Mamba': Mamba,
    'REBNCONV': REBNCONV
})
model.load_weights(main_model_path, by_name=True, skip_mismatch=True)

# Load other models for comparison
models = [
    load_model('./all_compare/u-segtime.h5', compile=False),
    load_model('./all_compare/cnn+bilstm.h5', compile=False),
    load_model('./all_compare/cnn_ecg.h5', compile=False),
    load_model('./all_compare/deeplab.h5', compile=False),
    load_model('./all_compare/encoder.h5', custom_objects={'InstanceNormalization': tfa.layers.InstanceNormalization}, compile=False),
    load_model('./all_compare/fcn.h5', compile=False),
    load_model('./all_compare/inception.h5', compile=False),
    load_model('./all_compare/resnet.h5', compile=False),
    load_model('./all_compare/xception.h5', compile=False),
    load_model('./all_compare/U-TSS.h5', compile=False, custom_objects={'weighted_categorical_crossentropy': custom_loss}),
]

model_names = [
    'U-SegTime', 'Dense-ECG', 'CNN', 'Deeplab', 'Encoder',
    'FCN', 'Inception', 'Resnet', 'Xception', 'U-TSS'
]

# Function to compute mIoU
def calculate_miou(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    TP = cm[1, 1]
    FP = cm[0, 1]
    FN = cm[1, 0]
    TN = cm[0, 0]
    iou_hvdc = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0
    iou_background = TN / (TN + FP + FN) if (TN + FP + FN) > 0 else 0
    miou = (iou_hvdc + iou_background) / 2
    return miou, iou_hvdc, iou_background

# Load test files
test_geo_files = os.listdir(test_geo_path)
test_label_files = os.listdir(test_label_path)

label_test = []
label_preds = [[] for _ in range(len(models))]
probabilities_hvdcs = [[] for _ in range(len(models))]

scaler = StandardScaler()

# Loop through test samples
for i in range(len(test_geo_files)):
    if test_geo_files[i] != test_label_files[i]:
        print("File names do not match.")
        break

    geo = np.load(test_geo_path + test_geo_files[i]).reshape(-1, 1)
    label = np.load(test_label_path + test_label_files[i])
    label_list = label.tolist()
    label_test.extend(label_list)

    geo = scaler.fit_transform(geo)
    geo = np.expand_dims(geo, axis=0)

    for idx, model in enumerate(models):
        pred = model.predict(geo, batch_size=1)
        pred_labels = [1.0 if p[1] >= 0.5 else 0.0 for p in pred[0]]
        label_preds[idx].extend(pred_labels)
        probabilities_hvdcs[idx].extend([p[1] for p in pred[0]])

    if i % 100 == 0:
        print(f"{i + 1}/{len(test_geo_files)} processed")

# Evaluation
plt.figure(figsize=(12, 9))
mpl.rcParams['xtick.labelsize'] = 16
mpl.rcParams['ytick.labelsize'] = 16

for idx, preds in enumerate(label_preds):
    acc = accuracy_score(label_test, preds)
    precision = precision_score(label_test, preds, zero_division=0)
    recall = recall_score(label_test, preds, zero_division=0)
    f1 = f1_score(label_test, preds, zero_division=0)
    auc = roc_auc_score(label_test, probabilities_hvdcs[idx])
    ap = average_precision_score(label_test, probabilities_hvdcs[idx])
    miou, iou_hvdc, iou_background = calculate_miou(label_test, preds)

    # Dice calculation
    conf_matrix = confusion_matrix(label_test, preds)
    dice_per_class = []
    for i in range(len(np.unique(label_test))):
        tp = conf_matrix[i, i]
        fp = conf_matrix[:, i].sum() - tp
        fn = conf_matrix[i, :].sum() - tp
        denom = 2 * tp + fp + fn
        dice = (2 * tp) / denom if denom != 0 else 0
        dice_per_class.append(dice)
    mdice = np.mean(dice_per_class)

    print(f"\nModel {model_names[idx]}:")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUC: {auc:.4f}")
    print(f"AP: {ap:.4f}")
    print(f"IoU HVDC: {iou_hvdc:.4f}, IoU Background: {iou_background:.4f}, mIoU: {miou:.4f}")
    for i, dice in enumerate(dice_per_class):
        print(f"Class {i}: Dice: {dice:.4f}")
    print(f"Mean Dice: {mdice:.4f}")

    fpr, tpr, _ = roc_curve(label_test, probabilities_hvdcs[idx])
    plt.plot(fpr, tpr, lw=2, label=f'{model_names[idx]}', zorder=idx)

plt.xlabel('False Positive Rate', fontsize=16)
plt.ylabel('True Positive Rate', fontsize=16)
plt.title('ROC Curve', fontsize=16)
plt.legend(loc="lower right", fontsize=16)
plt.savefig('./compare_jijian/2ROC.png')
plt.show()

# Precision-Recall Curve
plt.figure(figsize=(16, 9))
for idx, name in enumerate(model_names):
    precisions, recalls, _ = precision_recall_curve(label_test, probabilities_hvdcs[idx])
    plt.step(recalls, precisions, where='post', label=f'{name}')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.legend(loc="lower left")
plt.savefig('./compare_jijian/1PR_curve.png')
plt.show()
