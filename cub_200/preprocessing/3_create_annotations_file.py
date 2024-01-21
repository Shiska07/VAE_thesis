import os
import csv

def create_csv(root_folder, output_csv):
    
    with open(output_csv, 'w+', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['image_path', 'label'])  # Header

        for class_folder in os.listdir(root_folder):

	    # path from root folder to class folder
            class_path = os.path.join(root_folder, class_folder)

            if os.path.isdir(class_path):
                class_label, class_name = class_folder.split('.', 1)
                class_label = int(class_label)
                for image_file in os.listdir(class_path):
                    if image_file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):

			# we only join class name, image name is we don't want root folder in the name 
                        image_path = class_folder+'/'+image_file
                        csv_writer.writerow([image_path, class_label])

if __name__ == "__main__":
    root_folder = "cub200_cropped/test_cropped/"  # Replace with the path to your root folder
    output_csv = "cub200_cropped/test_cropped_annotations.csv"  # Specify the desired output CSV file name

    create_csv(root_folder, output_csv)

    print(f"CSV file '{output_csv}' created successfully.")
    
