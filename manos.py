#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 13:03:06 2021

@author: rpp
"""

import librosa
import warnings
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.stats as stats
import math


def features_normalize_min_max(features):
    mins = features.min(axis=0)
    maxs = features.max(axis=0)
    return (features - mins) / (maxs - mins)


def feature_stats(feature):
    feature_mean = feature.mean(axis=1)
    feature_std = feature.std(axis=1)
    feature_skewness = stats.skew(feature, axis=1)
    feature_kurtosis = stats.kurtosis(feature, axis=1)
    feature_median = np.median(feature, axis=1)
    feature_min = feature.min(axis=1)
    feature_max = feature.max(axis=1)

    return np.column_stack((feature_mean, feature_std, feature_skewness,
                            feature_kurtosis, feature_median, feature_min,
                            feature_max))


if __name__ == "__main__":
    plt.close('all')

    musicas = os.listdir('Music')
    musicas = sorted(musicas)

    # 2.1--- Load Top100 Features file
    top100_features_norm_csv = 'top100_features_norm.csv'
    if os.path.isfile(top100_features_norm_csv):
        top100_features_np_norm = np.genfromtxt(
            top100_features_norm_csv, delimiter=',')
    else:
        top100_features_csv = "Features/top100_features.csv"
        top100_features_np = np.genfromtxt(
            top100_features_csv, skip_header=1, delimiter=',')
        top100_features_np_norm = features_normalize_min_max(
            top100_features_np[:, 1:])
        np.savetxt(top100_features_norm_csv,
                   top100_features_np_norm, delimiter=',', fmt='%.6f')

    # 2.2--- librosa
    librosa_features_norm_csv = 'manos/librosa_features_norm_10.csv'

    warnings.filterwarnings('ignore')

    librosa_features_np = np.empty((0, 183))
    counter = 0
    total = len(musicas)
    for musica in musicas[:10]:
        counter += 1
        print("MRS em " + musica + " " + str(counter) + "/" + str(total))
        musica_path = "Music/" + str(musica)
        print(musica_path)

        y, fs = librosa.load(musica_path, sr=22050, mono=True)
        # print(y.shape)
        # print(fs)

        mfcc = librosa.feature.mfcc(y=y, sr=22050, n_mfcc=13)
        # print(mfcc.shape)
        spectral_centroid = librosa.feature.spectral_centroid(
            y=y, sr=22050)
        # print(spectral_centroid.shape)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(
            y=y, sr=22050)
        # print(spectral_bandwidth.shape)
        spectral_contrast = librosa.feature.spectral_contrast(
            y=y, sr=22050)
        # print(spectral_contrast.shape)
        spectral_flatness = librosa.feature.spectral_flatness(y=y)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=22050)
        # print(spectral_rolloff.shape)
        rms = librosa.feature.rms(y=y)
        # print(rms.shape)
        # print(rms[0,:].shape)
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y=y)
        # print(zero_crossing_rate.shape)
        tempo = librosa.beat.tempo(y=y, sr=22050)
        # print(tempo.shape)

        mfcc_stats = feature_stats(mfcc)
        spectral_centroid_stats = feature_stats(spectral_centroid)
        spectral_bandwidth_stats = feature_stats(spectral_bandwidth)
        spectral_contrast_stats = feature_stats(spectral_contrast)
        spectral_flatness_stats = feature_stats(spectral_flatness)

        print("FLATNESSSSSSSS")
        print(spectral_flatness_stats)
        spectral_rolloff_stats = feature_stats(spectral_rolloff)
        rms_stats = feature_stats(rms)
        zero_crossing_rate_stats = feature_stats(zero_crossing_rate)
        print("centroid shape: ", np.shape(spectral_centroid_stats))
        music_features = np.concatenate((mfcc_stats.flatten(),
                                         spectral_centroid_stats[0, :],
                                         spectral_bandwidth_stats[0, :],
                                         spectral_contrast_stats.flatten(),
                                         spectral_flatness_stats[0, :],
                                         spectral_rolloff_stats[0, :],
                                         rms_stats[0, :],
                                         zero_crossing_rate_stats[0, :],
                                         tempo
                                         ))

        librosa_features_np = np.vstack(
            (librosa_features_np, music_features))

    librosa_features_norm_np = features_normalize_min_max(
        librosa_features_np)
    np.savetxt(librosa_features_norm_csv,
               librosa_features_norm_np, delimiter=',', fmt='%.6f')
