import numpy as np

# Create a custom Transformer function
def custom_transformer(X):
    return np.square(X)

class Datacleaner():
    def __init__(self, data):
        self.data = data
    
    def drop_na(self):
        self.data.dropna(how='any', axis=0, inplace=True)
        return self.data
    
    def unnecessary_cols(self, cols):
        self.data.drop(cols, axis=1, inplace=True)
        return self.data
    
    def create_response(self, col):
        self.data[col] = self.data[col].apply(lambda x: 1 if x=='A' else 0)
        return self.data


