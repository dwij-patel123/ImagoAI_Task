import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from preprocess import process_data
import pandas as pd

path = 'data/TASK-ML-INTERN.csv'
(X,y),(X_scaled,y) = process_data(path)

def visualize():
    # visualizing raw data with respect to wavelength and reflectance
    plt.figure(figsize=(10, 5))
    plt.plot(X.columns, X.mean(), label='Average Reflectance')
    plt.xlabel("wavelength bands")
    plt.ylabel("reflectance value")
    plt.legend()
    plt.title("Spectral Graph")
    plt.show()

    # HeatMap for spectral Data
    plt.figure(figsize=(12, 6))
    sns.heatmap(X_scaled, cmap='coolwarm', xticklabels=False)
    plt.title("Spectral Reflectance Heatmap")
    plt.show()

    # Heatmap for DON Concentration
    df_concentration = pd.DataFrame({'Sample Index': range(len(y)), 'DON Concentration': y})
    plt.figure(figsize=(10, 5))
    sns.heatmap(df_concentration.set_index('Sample Index').T, cmap='inferno', cbar=True)
    plt.title("Heatmap of DON Concentration")
    plt.xlabel("Samples")
    plt.ylabel("DON Concentration")
    plt.show()

    # Perform PCA to get more insights of the data
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    # data converted from (500,448 ) to (500,2)
    plt.figure(figsize=(8, 6))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.6)
    plt.colorbar(label="DON conventration")
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.title("PCA projection")
    plt.show()

visualize()