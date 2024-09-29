from flask import Flask, render_template, request, jsonify
from kmeans import KMeansClustering
import numpy as np

app = Flask(__name__)
kmeans = None
method = None
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/initialize', methods=['POST'])
def initialize():
    global kmeans, method
    clusters = int(request.form['clusters'])
    method = request.form['method']
    kmeans = KMeansClustering(clusters, method)
    return jsonify({'status': 'initialized'})

@app.route('/step', methods=['POST'])
def step():
    global kmeans
    if kmeans:
        converged = kmeans.step()  # Run one step of K-Means
        data = kmeans.data.tolist()
        labels = kmeans.labels.tolist()
        centroids = kmeans.centroids.tolist()
        return jsonify({
            'data': data,
            'labels': labels,
            'centroids': centroids,
            'converged': converged  # Return if it's converged
        })
    return jsonify({'error': 'KMeans not initialized'}), 400


@app.route('/manual', methods=['POST'])
def manual_init():
  global kmeans, method
  if kmeans and method == "manual":
    kmeans.manual_init(request.json);
    return jsonify({'message': 'Centroids Manually Initialized'}), 200
  return jsonify({'error': 'kmeans not initialized'}), 400

@app.route('/converge', methods=['POST'])
def converge():
    global kmeans
    if kmeans:
        kmeans.run_to_convergence()  # Run till convergence
        return jsonify({
            'data': kmeans.data.tolist(),
            'labels': kmeans.labels.tolist(),
            'centroids': kmeans.centroids.tolist()
        })
    return jsonify({'error': 'KMeans not initialized'}), 400

@app.route('/new_data', methods=['POST'])
def new_data():
    global kmeans
    if kmeans:
        kmeans.generate_new_data()  # Generate new random data
        return jsonify({
            'data': kmeans.data.tolist(),
            'labels': kmeans.labels.tolist(),
            'centroids': kmeans.centroids.tolist()
        })
    return jsonify({'error': 'KMeans not initialized'}), 400

if __name__ == '__main__':
    app.run(debug=True)
