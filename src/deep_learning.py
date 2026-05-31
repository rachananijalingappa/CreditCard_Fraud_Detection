"""Deep learning models: MLP, Autoencoder, LSTM, 1D-CNN."""

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (Dense, Dropout, BatchNormalization, LSTM as LSTMLayer,
                                     Conv1D, MaxPooling1D, Flatten, Input)
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

from src.config import RANDOM_STATE


def train_mlp(X_train, y_train, X_test, y_test, evaluate_fn,
              all_results, all_preds, all_probs):
    """Train a Multi-Layer Perceptron (Neural Network).
    
    Returns:
        Trained MLP model.
    """
    n_features = X_train.shape[1]
    
    mlp = Sequential([
        Dense(128, activation='relu', input_shape=(n_features,)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    
    mlp.compile(optimizer=Adam(learning_rate=0.001),
                loss='binary_crossentropy', metrics=['accuracy'])
    
    start = time.time()
    history = mlp.fit(X_train, y_train, epochs=50, batch_size=512,
                      validation_data=(X_test, y_test),
                      callbacks=[EarlyStopping(patience=10, restore_best_weights=True)],
                      verbose=0)
    t = time.time() - start
    
    mlp_prob = mlp.predict(X_test, verbose=0).flatten()
    mlp_pred = (mlp_prob >= 0.5).astype(int)
    evaluate_fn('Neural Network', y_test, mlp_pred, mlp_prob, t,
                all_results, all_preds, all_probs)
    
    # Plot training curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(history.history['loss'], label='Train')
    axes[0].plot(history.history['val_loss'], label='Val')
    axes[0].set_title('Loss Curve', fontweight='bold')
    axes[0].legend()
    axes[1].plot(history.history['accuracy'], label='Train')
    axes[1].plot(history.history['val_accuracy'], label='Val')
    axes[1].set_title('Accuracy Curve', fontweight='bold')
    axes[1].legend()
    plt.tight_layout()
    plt.show()
    
    return mlp


def train_autoencoder(X_train_scaled, y_train, X_test_scaled, y_test,
                     evaluate_fn, all_results, all_preds, all_probs):
    """Train an Autoencoder on legitimate transactions, detect fraud as anomalies.
    
    Returns:
        Trained autoencoder model.
    """
    X_train_legit = X_train_scaled[y_train == 0]
    n_features = X_train_scaled.shape[1]
    
    ae_input = Input(shape=(n_features,))
    encoded = Dense(64, activation='relu')(ae_input)
    encoded = Dense(32, activation='relu')(encoded)
    encoded = Dense(14, activation='relu')(encoded)
    decoded = Dense(32, activation='relu')(encoded)
    decoded = Dense(64, activation='relu')(decoded)
    decoded = Dense(n_features, activation='linear')(decoded)
    
    autoencoder = Model(ae_input, decoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    
    start = time.time()
    autoencoder.fit(X_train_legit, X_train_legit, epochs=50, batch_size=512,
                    validation_split=0.1,
                    callbacks=[EarlyStopping(patience=10, restore_best_weights=True)],
                    verbose=0)
    t = time.time() - start
    
    # Reconstruction error = anomaly score
    reconstructed = autoencoder.predict(X_test_scaled, verbose=0)
    recon_error = np.mean((X_test_scaled - reconstructed) ** 2, axis=1)
    
    # Set threshold based on training data
    train_recon = autoencoder.predict(X_train_legit, verbose=0)
    train_error = np.mean((X_train_legit - train_recon) ** 2, axis=1)
    threshold = np.percentile(train_error, 97)
    
    ae_pred = (recon_error > threshold).astype(int)
    ae_prob = recon_error / recon_error.max()
    evaluate_fn('Autoencoder', y_test, ae_pred, ae_prob, t,
                all_results, all_preds, all_probs)
    
    # Plot reconstruction error
    plt.figure(figsize=(10, 5))
    plt.hist(recon_error[y_test == 0], bins=100, alpha=0.7, color='#2ecc71',
             label='Legitimate', density=True)
    plt.hist(recon_error[y_test == 1], bins=50, alpha=0.7, color='#e74c3c',
             label='Fraud', density=True)
    plt.axvline(threshold, color='black', linestyle='--', label=f'Threshold={threshold:.4f}')
    plt.title('Autoencoder Reconstruction Error', fontweight='bold')
    plt.legend()
    plt.show()
    
    return autoencoder


def train_lstm(X_train, y_train, X_test, y_test, evaluate_fn,
              all_results, all_preds, all_probs):
    """Train an LSTM model treating features as a sequence.
    
    Returns:
        Trained LSTM model.
    """
    n_features = X_train.shape[1]
    X_train_seq = np.array(X_train).reshape(-1, n_features, 1)
    X_test_seq = np.array(X_test).reshape(-1, n_features, 1)
    
    lstm = Sequential([
        LSTMLayer(64, input_shape=(n_features, 1), return_sequences=True),
        Dropout(0.3),
        LSTMLayer(32),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    lstm.compile(optimizer=Adam(learning_rate=0.001),
                 loss='binary_crossentropy', metrics=['accuracy'])
    
    start = time.time()
    lstm.fit(X_train_seq, y_train, epochs=30, batch_size=1024,
             validation_data=(X_test_seq, y_test),
             callbacks=[EarlyStopping(patience=5, restore_best_weights=True)],
             verbose=0)
    t = time.time() - start
    
    lstm_prob = lstm.predict(X_test_seq, verbose=0).flatten()
    lstm_pred = (lstm_prob >= 0.5).astype(int)
    evaluate_fn('LSTM', y_test, lstm_pred, lstm_prob, t,
                all_results, all_preds, all_probs)
    
    return lstm


def train_cnn(X_train, y_train, X_test, y_test, evaluate_fn,
             all_results, all_preds, all_probs):
    """Train a 1D-CNN model.
    
    Returns:
        Trained CNN model.
    """
    n_features = X_train.shape[1]
    X_train_seq = np.array(X_train).reshape(-1, n_features, 1)
    X_test_seq = np.array(X_test).reshape(-1, n_features, 1)
    
    cnn = Sequential([
        Conv1D(64, kernel_size=3, activation='relu', input_shape=(n_features, 1), padding='same'),
        BatchNormalization(),
        MaxPooling1D(2),
        Dropout(0.3),
        Conv1D(32, kernel_size=3, activation='relu', padding='same'),
        MaxPooling1D(2),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    
    cnn.compile(optimizer=Adam(learning_rate=0.001),
                loss='binary_crossentropy', metrics=['accuracy'])
    
    start = time.time()
    cnn.fit(X_train_seq, y_train, epochs=30, batch_size=1024,
            validation_data=(X_test_seq, y_test),
            callbacks=[EarlyStopping(patience=5, restore_best_weights=True)],
            verbose=0)
    t = time.time() - start
    
    cnn_prob = cnn.predict(X_test_seq, verbose=0).flatten()
    cnn_pred = (cnn_prob >= 0.5).astype(int)
    evaluate_fn('1D-CNN', y_test, cnn_pred, cnn_prob, t,
                all_results, all_preds, all_probs)
    
    return cnn
