# -*- coding: utf-8 -*-
'''
This script loads a trained model and applies it to a set of test samples.
It visualizes the detection results of HVDC events on the test set alongside the ground truth annotations.

Each plot displays:
- The raw geomagnetic signal
- The predicted probability of HVDC events
- The ground truth HVDC event regions

These visualizations correspond to **Figures 6, 7, and 10** in the manuscript.

'''
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn import preprocessing as prep
import tensorflow as tf
import tensorflow.keras.backend as K
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Load the model for this test
experiment_code = "FEexperiment"
# model = load_model('../image/image_deeplab/{}/bmodel.h5'.format(experiment_code), compile=False)
model = load_model('./u-hvdc_compare/u-segtime.h5', compile=False)
All_N_act_HVDC = 0  # Total number of actual HVDC events in the entire test set
All_N_cat_HVDC = 0  # Total number of predicted HVDC events in the entire test set
All_N_act_HVDC_good = 0  # Total number of actual HVDC events correctly predicted in the entire test set
All_N_cat_HVDC_good = 0  # Total number of predicted HVDC events that are correct in the entire test set


test_geo_path = '../../1418_test_geo/'  # Test data
test_label_path = '../../1418_test_label/'  # Test labels
geo_files = os.listdir(test_geo_path)
label_files = os.listdir(test_label_path)

import matplotlib as mpl
def plot_with_confidence_intervals(geo_data, pred_data, split_lst_x, split_lst_y, select, i, total):
    plt.rcParams['figure.figsize'] = (16.0, 10.0)
    # Set global font size
    mpl.rcParams['xtick.labelsize'] = 20  # x-axis tick label font size
    mpl.rcParams['ytick.labelsize'] = 20  # y-axis tick label font size
    title_name = f"Station code: {station_code}, Instrument code: {instrument_code}, Date: {date[4:6]}/{date[6:]}/{date[:4]}"
    print(title_name)
    # Left axis: plot geo_data and predicted probabilities
    fig, ax1 = plt.subplots()
    ax1.plot(geo_data, label='BACKGROUND', color='black')

    ax1.fill_between(range(len(pred_data)), 0, pred_data[:, 0], color='#2c73d2', alpha=1,
                     label='Probability of BACKGROUND')
    ax1.fill_between(range(len(pred_data)), 0, pred_data[:, 1], color='red', alpha=1,
                     label='Probability of HVDC')

    for v in range(len(split_lst_x)):
        ax1.plot(split_lst_x[v], split_lst_y[v], color="red", linewidth=1.0,
                 label='HVDC' if v == 0 else "")

    ax1.legend(loc='lower right', fontsize=20)
    ax1.set_xlabel('Time (minutes)', fontsize=20)
    ax1.set_ylabel('Value', fontsize=20)

    # Add right y-axis but hide ticks
    ax2 = ax1.twinx()
    ax2.set_yticks([])
    ax2.set_ylabel('Probability', fontsize=20, color='black', labelpad=20)

    # Save image
    plt.title(title_name, fontsize=20)
    plt.savefig("./results/plot{}/{}.png".format(experiment_code, select.split(".")[0]))
    plt.close()


# Iterate through test data and labels
if len(geo_files) == len(label_files):
    for i in range(len(geo_files)):
        # Extract station/instrument/date info from filename
        parts = geo_files[i].split('_')
        station_code = parts[0]
        instrument_code = parts[1]
        date = parts[3].split('.')[0]

        select = geo_files[i]
        if geo_files[i] == label_files[i]:
            a_geo = np.load(test_geo_path + select)
            a_seg = np.load(test_label_path + select)
            hvdc_x = [t3 for t3 in range(len(a_seg)) if a_seg[t3] == 1.0]

            len_lst = len(hvdc_x)
            t = 0
            split_lst_x_f = []
            tmp_lst = [hvdc_x[t]]
            while True:
                if t + 1 == len_lst:
                    break
                next_n = hvdc_x[t + 1]
                if hvdc_x[t] + 1 == next_n:
                    tmp_lst.append(next_n)
                else:
                    split_lst_x_f.append(tmp_lst)
                    tmp_lst = [next_n]
                t += 1

            split_lst_x_f.append(tmp_lst)
            split_lst_x = [event for event in split_lst_x_f]
            split_lst_y = [[a_geo[idx] for idx in event] for event in split_lst_x]

            K.clear_session()
            tf.compat.v1.reset_default_graph()

            a_geo_scaled = np.expand_dims(prep.scale(a_geo), axis=1)
            a_geo_scaled = np.expand_dims(a_geo_scaled, axis=0)
            tic = time.time()
            a_pred = model.predict(a_geo_scaled)
            toc = time.time()

            print('Elapsed time: ' + str(toc - tic) + ' seconds.')
            plot_with_confidence_intervals(a_geo, a_pred[0], split_lst_x, split_lst_y, select, i, len(geo_files))

# Compute and print statistical metrics (code remains unchanged)

            # Compute current sample metrics and add to totals
            list_real = split_lst_x  # List of actual HVDC events
            list_catch = []  # List of predicted HVDC events
            list_pred_over_half = []  # Predicted positions where HVDC probability > 0.5
            time_threshold = 5  # Ignore predicted HVDC shorter than this duration
            list_pred = a_pred[0].tolist()
            for t4 in range(len(list_pred)):
                prob_background = list_pred[t4][0]
                prob_hvdc = list_pred[t4][1]
                if prob_hvdc >= 0.5:
                    list_pred_over_half.append(t4)
            if len(list_pred_over_half) == 0:
                list_catch = []
            else:
                len_lst = len(list_pred_over_half)
                q = 0
                tmp_lst = [list_pred_over_half[q]]
                while True:
                    if q + 1 == len_lst:
                        break
                    next_n = list_pred_over_half[q + 1]
                    if list_pred_over_half[q] + 1 == next_n:
                        tmp_lst.append(next_n)
                    else:
                        list_catch.append(tmp_lst)
                        tmp_lst = [next_n]
                    q += 1
                list_catch.append(tmp_lst)
            list_catch_final = []
            for e in range(len(list_catch)):
                list_catch_final.append(list_catch[e])

            # Compute per-sample metrics and aggregate
            list_actual_good = []  # Actual events that were correctly predicted
            list_catch_good = []  # Predicted events that matched actual ones
            for d in range(len(list_real)):
                for f in range(len(list_catch_final)):
                    set_com = set(list_real[d]).intersection(set(list_catch_final[f]))
                    list_com = list(set_com)
                    if len(list_com) > 0 and list_real[d]:
                        if list_real[d] not in list_actual_good:
                            list_actual_good.append(list_real[d])
                        if list_catch_final[f] not in list_catch_good:
                            list_catch_good.append(list_catch_final[f])

            N_act_HVDC = len(list_real)
            N_cat_HVDC = len(list_catch_final)
            N_act_HVDC_good = len(list_actual_good)
            N_cat_HVDC_good = len(list_catch_good)

            # Add to final totals
            All_N_act_HVDC = All_N_act_HVDC + N_act_HVDC
            All_N_cat_HVDC = All_N_cat_HVDC + N_cat_HVDC
            All_N_act_HVDC_good = All_N_act_HVDC_good + N_act_HVDC_good
            All_N_cat_HVDC_good = All_N_cat_HVDC_good + N_cat_HVDC_good
            print("{}/{}".format(i + 1, len(geo_files)))
        else:
            print("Sample error, break")
            print("{}".format(select))
            break

# Start calculating final evaluation metrics
print("All test samples processed. Calculating final evaluation metrics...")
rate_A = All_N_act_HVDC_good / All_N_act_HVDC
rate_B = (All_N_act_HVDC - All_N_act_HVDC_good) / All_N_act_HVDC
print('Recall rate', rate_A, type(rate_A))
print('Miss rate', rate_B, type(rate_B))
if All_N_cat_HVDC == 0:
    print('Precision rate not computable', "Model predicted no HVDC")
    print('False alarm rate not computable', "Model predicted no HVDC")

else:
    rate_C = All_N_cat_HVDC_good / All_N_cat_HVDC
    rate_D = (All_N_cat_HVDC - All_N_cat_HVDC_good) / All_N_cat_HVDC
    print('Precision rate', rate_C, type(rate_C))
    print('False alarm rate', rate_D, type(rate_D))

print("Done")
