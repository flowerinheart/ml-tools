from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

def numeric(data, features):
    #scaling for no-tree models
    scaler = MinMaxScaler()
    scaler.fit(data[features])
    data[features] = scaler.transform(data[features])

    #scaler = StandardScaler()
