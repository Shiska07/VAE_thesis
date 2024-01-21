import os

def write_subdirectories_to_txt(folder_path, output_file):
    try:
        with open(output_file, 'w') as txt_file:
            subdirectories = [subdir for subdir in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, subdir))]
            txt_file.write('\n'.join(subdirectories))
        print(f"Subdirectories written to {output_file} successfully.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    folder_path = './cub200_cropped/train_cropped_augmented'

    output_file = "classnames.txt"  # Output file name

    write_subdirectories_to_txt(folder_path, output_file)
