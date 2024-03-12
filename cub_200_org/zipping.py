
import os
import zipfile

def create_zip(input_dir, output_zip):
    with zipfile.ZipFile(output_zip, 'w') as zipf:
        for root, _, files in os.walk(input_dir):
            for file in files:
                file_path = os.path.join(root, file)
                zipf.write(file_path, arcname=os.path.relpath(file_path, input_dir))

# Example usage:
input_dir = 'C:\\Users\\shisk\\Desktop\\Projects\\VAE_thesis\\cub_200_org\\data'
output_zip = 'C:\\Users\\shisk\\Desktop\\Projects\\VAE_thesis\\cub_200_org\\data.zip'
create_zip(input_dir, output_zip)