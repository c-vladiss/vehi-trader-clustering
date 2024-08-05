from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
from car_functions import (
    get_cars_data,
    preprocess_data,
    elbow_method,
    kmeans_clustering,
    umap_dbscan_clustering,
    get_closest_car_ids,
    get_price_stats
)

app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}})
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route('/', methods=['GET'])
def df():
    cars_df = get_cars_data()
    return jsonify(cars_df.to_dict())

@app.route('/kmeans_clusters', methods=['GET'])
def kmeans_clusters():
    cars_df = get_cars_data()
    cars_df_curated, cars_models_df = preprocess_data(cars_df)
    optimal_k = elbow_method(cars_df_curated)
    cars_df_curated = kmeans_clustering(cars_df_curated, optimal_k)
    return jsonify(cars_df_curated.to_dict())

@app.route('/umap_dbscan_clusters', methods=['GET'])
def umap_dbscan_clusters():
    cars_df = get_cars_data()
    cars_df_curated, cars_models_df = preprocess_data(cars_df)
    umap_df = umap_dbscan_clustering(cars_df_curated)
    return jsonify(umap_df.to_dict())

@app.route('/closest_cars/<car_id>', methods=['GET'])
def closest_cars(car_id):
    cars_df = get_cars_data()
    cars_df_curated, cars_models_df = preprocess_data(cars_df)
    features_for_clustering = ["kilometers", "horsePower", "price", "engineSize", "transmission", "fuelType", "gearbox", "year", "bodyType"]
    closest_car_ids = get_closest_car_ids(cars_df_curated, cars_models_df, car_id, features_for_clustering)
    return jsonify(list(closest_car_ids))

@app.route('/price_prediction/<car_id>', methods=['GET'])
def price_prediction(car_id):
    cars_df = get_cars_data()
    cars_df_curated, cars_models_df = preprocess_data(cars_df)
    features_for_clustering = ["kilometers", "horsePower", "price", "engineSize", "transmission", "fuelType", "gearbox", "year", "bodyType"]
    closest_car_ids = get_closest_car_ids(cars_df_curated, cars_models_df, car_id, features_for_clustering)
    closest_cars_indexes = cars_models_df[cars_models_df['_id'].astype(str).isin(closest_car_ids)].index
    mean_price, max_mean_price, min_mean_price = get_price_stats(cars_df, closest_cars_indexes)
    return jsonify({'mean_price': mean_price, 'max_mean_price': max_mean_price, 'min_mean_price': min_mean_price})


@app.route('/closest_cars_umap/<int:specific_car_index>', methods=['GET'])
def closest_cars_umap(specific_car_index):
    cars_df = get_cars_data()
    cars_df_curated, cars_models_df = preprocess_data(cars_df)
    umap_df = umap_dbscan_clustering(cars_df_curated)
    closest_car_ids = get_closest_car_ids(umap_df, cars_models_df, specific_car_index, ['UMAP1', 'UMAP2', 'UMAP3'])
    closest_cars_details = cars_models_df[cars_models_df['_id'].astype(str).isin(closest_car_ids)]
    return jsonify(closest_cars_details.to_dict(orient='records'))


@app.route('/get_price_estimate', methods=['GET'])
def get_price_estimate():
    car_details = request.args.to_dict()
    cars_df = get_cars_data()
    new_car_df = pd.DataFrame([car_details])
    new_car_df = new_car_df.astype(cars_df.dtypes.to_dict())
    cars_df = pd.concat([cars_df, new_car_df], ignore_index=True)
    cars_df_curated, cars_models_df = preprocess_data(cars_df)
    new_car_index = cars_df_curated.index[-1]
    new_car_id = cars_models_df.loc[new_car_index, '_id']
    print(cars_df.iloc[-1])
    print(cars_models_df.iloc[-1])
    print(cars_df.columns)
    features_for_clustering = ["kilometers", "horsePower", "price", "engineSize", "transmission", "fuelType", "gearbox", "year", "bodyType"]
    closest_car_ids = get_closest_car_ids(cars_df_curated, cars_models_df, new_car_id, features_for_clustering)
    closest_cars_indexes = cars_models_df[cars_models_df['_id'].astype(str).isin(closest_car_ids)].index
    mean_price, max_mean_price, min_mean_price = get_price_stats(cars_df, closest_cars_indexes)
    return jsonify({'mean_price': mean_price, 'max_mean_price': max_mean_price, 'min_mean_price': min_mean_price})


if __name__ == '__main__':
    app.run(debug=True)