import stow
from tqdm import tqdm

#########################
# Preprocessing Dataset #
#########################

# Construct the file path of the dataset "Dataset/"
dataset_path = stow.join('Dataset')

# Construct the file path "Dataset/words.txt" and set to read-only mode
words = open(stow.join(dataset_path, "words.txt"), "r").readlines()

# Define three variables
# dataset ->  A list of lists, each inner list contain relative file path and transcription of the image
# vocab -> A set which only contains unique words/transcriptions in the dataset
# max_len -> The maximum length of the labels.
dataset, vocab, max_len = [], set(), 0

# Loop through word.txt
# Sample Format: a01-000u-00-00 ok 154 1 408 768 27 51 AT A
# info[0] -> word id
# info[1] -> result of word segmentation (ok/err)
# info[2] -> graylevel
# info[3] -> number of components for this word    
# info[-1] -> the transcription for this word
for line in tqdm(words):
    # Ignore comments 
    if line.startswith("#"):
        continue

    # Split the line by space and arrange into a list 
    info = line.split(" ")

    # Ignore result of word segmentation which can be bad
    if info[1] == "err":
        continue

    # Sample directory of an image: Dataset/Image/a01/a01-000u/a01-000u-00-00.png
    identifier = info[0][:3]    # Extract the first three identifier
    form = info[0][:8]          # Extract the form of the image
    img_name = info[0] + ".png" # Extract the image name

    # Construct the file path of the images in the dataset "Dataset/Image/../../.."
    rel_path = stow.join(dataset_path, "Image", identifier, form, img_name) 
    if not stow.exists(rel_path):
        continue
    
    # Extract the transcription of the word and remove trailing characters to the right
    transcription = info[-1].rstrip('\n')

    # Update the three variables defined earlier
    dataset.append([rel_path, transcription])
    vocab.update(list(transcription))
    max_len = max(max_len, len(transcription))