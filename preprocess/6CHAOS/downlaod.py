import requests
from tqdm import tqdm

def download_file(url, output_path):
    # Send a HTTP request to the specified URL
    response = requests.get(url, stream=True)
    
    # Get the total file size in bytes
    total_size = int(response.headers.get('content-length', 0))
    
    # Open the output file in binary write mode
    with open(output_path, 'wb') as file, tqdm(
        desc=output_path,
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as progress_bar:
        # Iterate over the response data in chunks
        for data in response.iter_content(chunk_size=1024):
            # Write the chunk to the file
            file.write(data)
            # Update the progress bar
            progress_bar.update(len(data))

# URL of the file to download
# url = "https://zenodo.org/records/3431873/files/CHAOS_Train_Sets.zip?download=1"

# test:
url = "https://zenodo.org/records/3431873/files/CHAOS_Test_Sets.zip?download=1"

# Output path where the file will be saved
# output_path = "/home/local/ASURITE/longchao/Desktop/project/GE_health/SegmentAsYouWish/preprocess/6CHAOS/CHAOS_Train_Sets.zip"
output_path = "/home/local/ASURITE/longchao/Desktop/project/GE_health/SegmentAsYouWish/preprocess/6CHAOS/CHAOS_Test_Sets.zip"

# Download the file with progress
download_file(url, output_path)
