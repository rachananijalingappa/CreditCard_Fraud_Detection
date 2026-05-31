"""Centralised configuration and constants."""

import numpy as np
import tensorflow as tf
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

# Constants
RANDOM_STATE = 42
DATA_PATH = 'Data/creditcard.csv'
TEST_SIZE = 0.2
PCA_FEATURES = [f'V{i}' for i in range(1, 29)]
COLORS = ['#2ecc71', '#e74c3c']

# Reproducibility
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

# Display settings
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('viridis')
