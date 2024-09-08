import json
import random

# input yor file path: 
input_file_path = "/Users/danielsmith/Documents/1-RL/ASU/research/24GE-segment/SegmentAsYouWish/cleaned_augmented_data_rui_full.json"

def sample_data(data):
    """"
    random sample, iterate over every key
    """
    for key, values in data.items():
        if isinstance(values, list) and all(isinstance(value, str) for value in values):
            random_sample = random.choice(values)
            print(f"Key: {key}, Random Sample: {random_sample}")


def print_all_keys(data):
    """
    print all of the keys
    """    
    keys = data.keys()
    keys = list(keys)
    print("Length of keys:", len(keys))
    print("Keys:", list(keys))

def sample_n_descriptions(data, key, n):
    """
    sample based on key and parameter n
    """
    if key in data and isinstance(data[key], list):
        sampled_descriptions = random.sample(data[key], min(n, len(data[key])))
        print(f"Key: {key}, Sampled Descriptions: {sampled_descriptions}")
    else:
        print(f"Key '{key}' not found or value is not a list.")


with open(input_file_path, 'r') as infile:
    data = json.load(infile)

# Call the function to sample and print data
sample_data(data)

# Print all keys
print_all_keys(data)

# example, sample 3 descriptions from the "liver" key
sample_n_descriptions(data, "liver", 3)