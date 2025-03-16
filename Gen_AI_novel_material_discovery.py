# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pymatgen import Composition
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from matminer.featurizers.conversions import StrToComposition
from matminer.featurizers.composition import ElementProperty
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Input, Lambda
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam

# Step 1: Load the dataset
data = pd.read_csv("https://raw.githubusercontent.com/materialsvirtuallab/matminer/master/matminer/datasets/materials_project_multitarget.csv")

# Step 2: Data Preprocessing
data = StrToComposition().featurize_dataframe(data, "formula")
ep_featurizer = ElementProperty.from_preset("magpie")
data = ep_featurizer.featurize_dataframe(data, col_id="composition")
data = data.dropna()

# Step 3: Define Features and Targets
X = data.drop(columns=["formula", "composition", "band_gap", "formation_energy_per_atom", "bulk_modulus"])
y = data[["band_gap", "formation_energy_per_atom", "bulk_modulus"]]

# Step 4: Train a Predictive Model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 5: Build a Variational Autoencoder (VAE) for Material Generation
input_dim = X.shape[1]
latent_dim = 10

# Encoder
inputs = Input(shape=(input_dim,))
h = Dense(64, activation="relu")(inputs)
z_mean = Dense(latent_dim)(h)
z_log_var = Dense(latent_dim)(h)

# Sampling function
def sampling(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

z = Lambda(sampling)([z_mean, z_log_var])

# Decoder
decoder_h = Dense(64, activation="relu")
decoder_mean = Dense(input_dim, activation="sigmoid")
h_decoded = decoder_h(z)
x_decoded_mean = decoder_mean(h_decoded)

# VAE model
vae = Model(inputs, x_decoded_mean)

# Loss function
reconstruction_loss = K.sum(K.binary_crossentropy(inputs, x_decoded_mean), axis=-1)
kl_loss = -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
vae_loss = K.mean(reconstruction_loss + kl_loss)
vae.add_loss(vae_loss)

# Compile and train VAE
vae.compile(optimizer=Adam(learning_rate=0.001))
vae.fit(X_train, X_train, epochs=50, batch_size=32, validation_data=(X_test, X_test))

# Step 6: Generate New Materials
# Sample from the latent space
n_samples = 10
z_sample = np.random.normal(size=(n_samples, latent_dim))
generated_materials = decoder_mean(decoder_h(z_sample))

# Step 7: Predict Properties of Generated Materials
generated_properties = model.predict(generated_materials)

# Step 8: Active Learning
# Recommend materials with optimal properties (e.g., bandgap ~1.5 eV)
optimal_materials = generated_materials[(generated_properties[:, 0] > 1.0) & (generated_properties[:, 0] < 1.7)]
print("Optimal Materials for Solar Cells:")
print(optimal_materials)

# Step 9: Visualize Results
# Plot latent space
plt.figure(figsize=(8, 6))
plt.scatter(z_mean[:, 0], z_mean[:, 1], c=y_train["band_gap"], cmap="viridis")
plt.colorbar(label="Bandgap (eV)")
plt.xlabel("Latent Dimension 1")
plt.ylabel("Latent Dimension 2")
plt.title("Latent Space of Material Compositions")
plt.show()
