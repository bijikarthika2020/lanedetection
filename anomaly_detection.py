import random
import matplotlib.pyplot as plt
import numpy as np

# Simulating the Data Stream
def generate_data_stream(n_points=1000, anomaly_rate=0.05):
    """
    Simulate a data stream with anomalies.
    :param n_points: Number of points in the stream.
    :param anomaly_rate: Percentage of data points that are anomalies.
    :return: List of data points representing the stream.
    """
    data = []
    for i in range(n_points):
        # Normal data follows a sine wave with some random noise
        point = np.sin(i * 0.01) + random.uniform(-0.2, 0.2)
        # Introduce anomalies randomly
        if random.random() < anomaly_rate:
            point += random.uniform(3, 5)  # Large spike for anomaly
        data.append(point)
    return data

# Anomaly Detection Algorithm (Z-score)
def detect_anomalies(data, threshold=3):
    """
    Detect anomalies in a data stream using Z-score method.
    :param data: Data stream list.
    :param threshold: Z-score threshold for flagging anomalies.
    :return: List of indices where anomalies are detected.
    """
    anomalies = []
    mean = np.mean(data)
    std_dev = np.std(data)
    
    for i, point in enumerate(data):
        z_score = (point - mean) / std_dev
        if abs(z_score) > threshold:
            anomalies.append(i)
    return anomalies

# Optimization of the algorithm by using a sliding window
def optimized_detect_anomalies(data, window_size=50, threshold=3):
    """
    Detect anomalies using a sliding window Z-score calculation.
    :param data: Data stream list.
    :param window_size: Number of points for the moving average window.
    :param threshold: Z-score threshold for flagging anomalies.
    :return: List of indices where anomalies are detected.
    """
    anomalies = []
    for i in range(window_size, len(data)):
        window = data[i-window_size:i]
        mean = np.mean(window)
        std_dev = np.std(window)
        
        z_score = (data[i] - mean) / std_dev
        if abs(z_score) > threshold:
            anomalies.append(i)
    
    return anomalies

# Visualization
def visualize_data_stream(data, anomalies):
    """
    Visualize the data stream and highlight the detected anomalies.
    :param data: Data stream list.
    :param anomalies: List of indices where anomalies were detected.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(data, label='Data Stream')
    plt.scatter(anomalies, [data[i] for i in anomalies], color='red', label='Anomalies', marker='x')
    plt.title('Data Stream with Anomalies')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.show()

# Main Execution
if __name__ == "__main__":
    # Generate data stream
    data_stream = generate_data_stream(n_points=1000, anomaly_rate=0.05)
    
    # Detect anomalies (optimized version)
    anomalies = optimized_detect_anomalies(data_stream, window_size=50, threshold=3)
    
    # Visualize the result
    visualize_data_stream(data_stream, anomalies)
