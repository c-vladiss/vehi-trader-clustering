import matplotlib.pyplot as plt
import pandas as pd
import umap
from pymongo import MongoClient
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

client = MongoClient("db uri")
client.admin.command("ping")
db = client["vehi-trader-db"]
col = db['cars']

cars_cursor = col.find()
cars_list = list(cars_cursor)
cars_df = pd.DataFrame(cars_list)

cars_models_df = cars_df[["_id", "make", "model", "version", "generation"]]

cars_df_curated = cars_df.drop(columns=["_id", "make", "model", "version","generation", "photos"])

def extract_and_verify(value):
    extracted_value = pd.to_numeric(value, errors='coerce')
    return 1 if pd.isna(extracted_value) else int(extracted_value)

def extract_number(df, column_name):
    if column_name in df.columns:
        df[column_name] = df[column_name].str.replace(' ', '').str.extract('(\d+)')[0].apply(extract_and_verify)

def value_to_number(values):
    return {value: idx + 1 for idx, value in enumerate(values)}

culori_dict = value_to_number(cars_df_curated["color"].unique())
tipuri_caroserii_dict = value_to_number(cars_df_curated["bodyType"].unique())
transmisii_dict = value_to_number(cars_df_curated["transmission"].unique())
combustibil_dict = value_to_number(cars_df_curated["fuelType"].unique())
cutii_viteze_dict = value_to_number(cars_df_curated["gearbox"].unique())

cars_df_curated['color'] = cars_df_curated['color'].map(culori_dict)
cars_df_curated['bodyType'] = cars_df_curated['bodyType'].map(tipuri_caroserii_dict)
cars_df_curated['transmission'] = cars_df_curated['transmission'].map(transmisii_dict)
cars_df_curated['fuelType'] = cars_df_curated['fuelType'].map(combustibil_dict)
cars_df_curated['gearbox'] = cars_df_curated['gearbox'].map(cutii_viteze_dict)

extract_number(cars_df_curated, "emissionStandard")

columns_to_standardize = ["kilometers", "horsePower", "price", "engineSize", "transmission", "year", "bodyType"]
cars_df_curated = cars_df_curated.dropna()
scaler = StandardScaler()
cars_df_curated[columns_to_standardize] = scaler.fit_transform(cars_df_curated[columns_to_standardize])


def elbow_method(data, max_clusters=10):
    distortions = []
    for i in range(1, max_clusters + 1):
        km = KMeans(n_clusters=i, init='k-means++', n_init=10, max_iter=300, random_state=0)
        km.fit(data)
        distortions.append(km.inertia_)
    second_derivative = []
    for j in range(1, len(distortions) - 1):
        second_derivative.append(distortions[j - 1] - 2 * distortions[j] + distortions[j + 1])

    # Find the index of the maximum value in the second derivative
    optimal_k_index = second_derivative.index(max(second_derivative)) + 1
    optimal_k = optimal_k_index + 1  # Add 1 because of zero-indexing

    return optimal_k

k = elbow_method(cars_df_curated)
kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, max_iter=300, random_state=0)
cars_df_curated['Cluster'] = kmeans.fit_predict(cars_df_curated[columns_to_standardize])

def plot_clusters_2d(df, x, y):
    plt.figure(figsize=(15, 8))
    plt.scatter(df[x], df[y], c=df['Cluster'], cmap='viridis')
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title('Clusters of Cars')
    plt.colorbar(label='Cluster')
    plt.show()

def plot_clusters_3d(df, x, y, z):
    fig = plt.figure(figsize=(15, 8))
    ax = plt.axes(projection='3d')
    scatter = ax.scatter3D(df[x], df[y], df[z], c=df['Cluster'], cmap='viridis')
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_zlabel(z)
    ax.set_title('Clusters of Cars')
    fig.colorbar(scatter, ax=ax, label='Cluster')
    plt.show()


specific_car_index = 123
features_for_clustering = ["kilometers", "horsePower", "price", "engineSize", "transmission", "fuelType", "gearbox", "year", "bodyType"]
specific_car_features = cars_df_curated.loc[specific_car_index, features_for_clustering].values.reshape(1, -1)


def get_closest_car_ids(cars_df_curated, cars_models_df, specific_car_index, features_for_clustering, num_closest=10):
    specific_car_features = cars_df_curated.loc[specific_car_index, features_for_clustering].values.reshape(1, -1)
    distances = pairwise_distances(cars_df_curated[features_for_clustering], specific_car_features, metric='euclidean')
    distances_df = pd.DataFrame({'Distance': distances.flatten(), 'OriginalIndex': cars_df_curated.index})
    closest_cars_indices = distances_df.sort_values(by='Distance').head(num_closest)['OriginalIndex'].tolist()
    ids = cars_models_df.loc[closest_cars_indices]['_id'].values.tolist()
    return ids

closest_car_ids = get_closest_car_ids(cars_df_curated, cars_models_df, specific_car_index, features_for_clustering)



def get_price_stats(cars_df, car_ids):
    closest_cars_df = cars_df.loc[car_ids]
    mean_price = closest_cars_df['price'].mean()
    max_mean_price = mean_price * 1.1
    min_mean_price = mean_price * 0.9
    return mean_price, max_mean_price, min_mean_price
n_neighbors = 15
min_dist = 0.1
umap_3d = umap.UMAP(n_components=3, n_neighbors=n_neighbors, min_dist=min_dist, random_state=432)
umap_result = umap_3d.fit_transform(cars_df_curated.drop(columns=['Cluster']))

dbscan = DBSCAN(eps=0.5, min_samples=5)
# Creating a DataFrame with the UMAP results
umap_df = pd.DataFrame(umap_result, columns=['UMAP1', 'UMAP2', 'UMAP3'])
umap_df['Cluster'] = dbscan.fit_predict(umap_result)

# Plotting the UMAP results using the provided plot_clusters_3d function
plot_clusters_2d(cars_df_curated, 'kilometers', 'price')
plot_clusters_3d(cars_df_curated, 'kilometers', 'price', 'year')
plot_clusters_3d(umap_df, 'UMAP1', 'UMAP2', 'UMAP3')

closest_car_ids_2 = get_closest_car_ids(umap_df, cars_models_df, specific_car_index, ['UMAP1', 'UMAP2', 'UMAP3'], num_closest=10)
closest_car_ids_2 = [str(oid) for oid in closest_car_ids_2]
closest_cars_details_2 = cars_models_df[cars_models_df['_id'].astype(str).isin(closest_car_ids_2)]

closest_car_ids = [str(oid) for oid in closest_car_ids]
closest_cars_details = cars_models_df[cars_models_df['_id'].astype(str).isin(closest_car_ids)]

print("veriunea clasica:")
print (closest_cars_details[['make', 'model', 'version']])
print("veriunea cu UMAP:")
print (closest_cars_details_2[['make', 'model', 'version']])
print("masina cautata:")
print(cars_df.iloc[specific_car_index])
