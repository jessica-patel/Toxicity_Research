import socket
import tensorflow as tf
import pandas as pd
from tensorflow.keras.layers import TextVectorization

host = "127.0.0.1"
port = 5001
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect((host, port))

# dota 2
MAX_FEATURES = 72763
# jigsaw
# MAX_FEATURES = 200000

df = pd.read_csv('../Datasets/dotaAndCyber.csv')

# dota 2
X = df['message'].astype(str)

vectorizer = TextVectorization(max_tokens=MAX_FEATURES,
                               output_sequence_length=1800,
                               output_mode='int')
vectorizer.adapt(X.values)

model = tf.keras.models.load_model('../TrainedModels/dotaAndCyberEpoch8.keras')


def GetRatings(sampleComment):
    sampleComment = tf.expand_dims(sampleComment, 0)
    result = model.predict(sampleComment)[0]
    return f"{result[0]}, {result[1]}, {result[2]}"


while True:
    receivedData = sock.recv(1024).decode("UTF-8")    
    data = GetRatings(receivedData)
    sock.sendall(data.encode("UTF-8"))
