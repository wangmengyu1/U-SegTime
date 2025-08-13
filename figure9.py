# -*- coding: utf-8 -*-
"""
This script compares the detection results of multiple models on the same test sample.
Each figure shows:
- The raw geomagnetic signal
- Ground truth HVDC event segments
- Predicted event probabilities from all models (as confidence bands)

The comparison visualizations correspond to **Figure 9** in the manuscript.
"""

import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import tensorflow_addons as tfa
from tensorflow.keras.models import load_model
from sklearn import preprocessing as prep
import matplotlib.gridspec as gridspec

# Set GPU configuration
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

# Set global font
mpl.rcParams['font.family'] = 'Times New Roman'

# Load model files
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
    load_model('./all_compare/U-TSS.h5', compile=False),
]

model_names = [
    'U-SegTime',
    'Dense-ECG',
    'CNN',
    'Deeplab',
    'Encoder',
    'FCN',
    'Inception',
    'Resnet',
    'Xception',
    'U-TSS',
]

# Set test data and label directories
test_geo_path = './sample_plot/data/'
test_label_path = './sample_plot/label/'

geo_files = sorted(os.listdir(test_geo_path))
label_files = sorted(os.listdir(test_label_path))

# Plotting function
def plot_with_confidence_intervals(geo_data, pred_list, split_lst_x, split_lst_y, select, model_names, model_order, station_code, instrument_code, date):
    num_models = len(pred_list)
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(num_models + 1, 1, height_ratios=[2] + [1] * num_models, hspace=0.6)

    # Main signal plot
    title = f"Station Code: {station_code}, Instrument Code: {instrument_code}, Date: {date[:4]}/{date[4:6]}/{date[6:]}"
    ax = plt.subplot(gs[0])
    ax.set_title(title, fontsize=14)
    ax.plot(geo_data, label='BACKGROUND', color='gray')
    ax.set_ylabel('Value', fontsize=12)

    for v in range(len(split_lst_x)):
        ax.plot(split_lst_x[v], split_lst_y[v], color="red", linewidth=1.0, label='HVDC' if v == 0 else "")
    ax.legend(loc='center left', bbox_to_anchor=(1.01, 0.5), fontsize=12)
    ax.grid(True)

    # Model prediction plots
    for idx, model_idx in enumerate(model_order):
        pred_data = pred_list[model_idx]
        ax = plt.subplot(gs[idx + 1])
        plt.subplots_adjust(right=0.8)

        pred_data[:, 0] = np.clip(pred_data[:, 0], 0, 1)
        pred_data[:, 1] = np.clip(pred_data[:, 1], 0, 1)
        ax.fill_between(range(len(pred_data)), 0, pred_data[:, 0], color='#0ddbf5', alpha=1, label='Background')
        ax.fill_between(range(len(pred_data)), 0, pred_data[:, 1], color='#d15254', alpha=1, label='HVDC')
        ax.set_ylim(0, 1)

        # Model name annotation
        ax.text(-0.02, 0.5, model_names[model_idx], fontsize=12, ha='right', va='center', transform=ax.transAxes)

        if idx == 0:
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles, labels, loc='center left', bbox_to_anchor=(1.01, 0.5), fontsize=8)
        ax.grid(True)

    ax.set_xlabel('Time (minutes)', fontsize=12)

    # Save figure
    os.makedirs('./results/plot_model_comparison/', exist_ok=True)
    plt.savefig(f"./results/plot_model_comparison/{select.split('.')[0]}.png")
    plt.close()

# Iterate through all test samples
if len(geo_files) == len(label_files):
    for i in range(len(geo_files)):
        if geo_files[i] != label_files[i]:
            continue

        parts = geo_files[i].split('_')
        station_code = parts[0]
        instrument_code = parts[1]
        date = parts[3].split('.')[0]

        select = geo_files[i]
        geo_data = np.load(os.path.join(test_geo_path, select))
        label_data = np.load(os.path.join(test_label_path, select))
        hvdc_x = [t for t in range(len(label_data)) if label_data[t] == 1.0]

        # Extract HVDC event segments
        split_lst_x_f = []
        if hvdc_x:
            tmp = [hvdc_x[0]]
            for idx in range(1, len(hvdc_x)):
                if hvdc_x[idx] == hvdc_x[idx - 1] + 1:
                    tmp.append(hvdc_x[idx])
                else:
                    split_lst_x_f.append(tmp)
                    tmp = [hvdc_x[idx]]
            split_lst_x_f.append(tmp)
        split_lst_x = split_lst_x_f
        split_lst_y = [[geo_data[idx] for idx in seg] for seg in split_lst_x]

        # Model predictions
        pred_list = []
        for model in models:
            geo_scaled = prep.scale(geo_data)
            geo_scaled = np.expand_dims(geo_scaled, axis=(0, 2))  # shape: (1, length, 1)
            pred = model.predict(geo_scaled)
            pred_list.append(pred[0])

        model_order = list(range(len(models)))  # default order
        plot_with_confidence_intervals(geo_data, pred_list, split_lst_x, split_lst_y, select, model_names, model_order, station_code, instrument_code, date)
else:
    print("Mismatch in number of input and label files.")
