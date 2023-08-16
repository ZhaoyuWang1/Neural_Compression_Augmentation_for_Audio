import os
import json

def generate_json(data_folder_path):
    # Path to the folder containing WAV files
    #data_folder_path = "example"

    # Get a list of WAV file names in the folder
    filenames = [file for file in os.listdir(data_folder_path) if file.endswith(".wav")]
    quarter_length = len(filenames) // 4
    split_1 = filenames[:quarter_length]
    split_2 = filenames[quarter_length:quarter_length*2]
    split_3 = filenames[quarter_length*2:quarter_length*3]
    split_4 = filenames[quarter_length*3:]
    # Generate the JSON data
    data_1 = [{"wav": os.path.join(data_folder_path, wav_file_1), "caption": ""} for wav_file_1 in split_1]
    json_data_1 = {"data": data_1}

    data_2 = [{"wav": os.path.join(data_folder_path, wav_file_2), "caption": ""} for wav_file_2 in split_2]
    json_data_2 = {"data": data_2}

    data_3 = [{"wav": os.path.join(data_folder_path, wav_file_3), "caption": ""} for wav_file_3 in split_3]
    json_data_3 = {"data": data_3}

    data_4 = [{"wav": os.path.join(data_folder_path, wav_file_4), "caption": ""} for wav_file_4 in split_4]
    json_data_4 = {"data": data_4}

    data_4 = [{"wav": os.path.join(data_folder_path, wav_file_4), "caption": ""} for wav_file_4 in split_4]
    json_data_4 = {"data": data_4}

    # Write the JSON data to a file
    json_filename_1 = "dataset_1.json"
    json_filename_2 = "dataset_2.json"
    json_filename_3 = "dataset_3.json"
    json_filename_4 = "dataset_4.json"
    with open(json_filename_1, "w") as json_file:
        json.dump(json_data_1, json_file, indent=4)

    with open(json_filename_2, "w") as json_file:
        json.dump(json_data_2, json_file, indent=4)
    
    with open(json_filename_3, "w") as json_file:
        json.dump(json_data_3, json_file, indent=4)

    with open(json_filename_4, "w") as json_file:
        json.dump(json_data_4, json_file, indent=4)
    print("JSON files generated successfully.")
    # Specify the directory to search

data_folder_path = r"/rds/general/user/zw1222/home/debug/SSL_audio/data/audioset/unbalanced_train_segments"
generate_json(data_folder_path=data_folder_path)