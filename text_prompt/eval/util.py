import pickle

def get_pickle(path=None):
    # Load the combined dictionary from the file
    with open(path, 'rb') as file:
        loaded_dict = pickle.load(file)
    
    return loaded_dict