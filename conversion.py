from pydub import AudioSegment
import os
import ffmpeg
# Input and output directories
input_directory = 'C:\\Users\\Harold\\Desktop\\data_audio\\FSD50K.ground_truth\\sample'
output_directory = 'C:\\Users\\Harold\\Desktop\\data_audio\\FSD50K.ground_truth\\sample\\output'

# Ensure the output directory exists
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Compress to MP3
def compress_to_mp3(input_path, bitrate="192k"):
    audio = AudioSegment.from_wav(input_path)
    output_path_mp3 = os.path.join(output_directory, os.path.basename(input_path).replace(".wav", ".mp3"))
    print(output_path_mp3)
    audio.export(output_path_mp3, format="mp3", bitrate=bitrate)
    return output_path_mp3

# Convert back to WAV
def convert_to_wav(mp3_path):
    audio = AudioSegment.from_mp3(mp3_path)
    output_path_wav = os.path.join(output_directory, os.path.basename(mp3_path).replace(".mp3", ".wav"))
    audio.export(output_path_wav, format="wav")
    return output_path_wav

# Delete the MP3 file
def delete_file(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"{file_path} deleted successfully")
    else:
        print(f"{file_path} not found")

# Process all WAV files in the directory
for filename in os.listdir(input_directory):
    if filename.endswith(".wav"):
        input_path = os.path.join(input_directory, filename)
        print(input_path)
        mp3_path = compress_to_mp3(input_path)
        wav_path = convert_to_wav(mp3_path)
        delete_file(mp3_path)
        print(f"{filename} compressed and converted back to WAV at {wav_path}")