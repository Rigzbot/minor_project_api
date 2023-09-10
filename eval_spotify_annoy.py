from annoy import AnnoyIndex
from deepface import DeepFace
import pickle
import os


# Load embeddings file
with open("spotify_annoy/embeddings.pkl", "rb") as file:
    representations = pickle.load(file)

embedding_size = 2622
t = AnnoyIndex(embedding_size, 'euclidean')
t.load('spotify_annoy/custom_vgg_annoy_model.ann')

# Threshold for classifying images as "unknown"
threshold = 0.40

# Function to classify an image
def classify_image(embedding, threshold, n_neighbours):
    # Query the Annoy index to find the nearest neighbor's index and distance
    nearest_neighbor_idx, nearest_neighbor_distance = t.get_nns_by_vector(embedding, n_neighbours, include_distances=True)
    
    if nearest_neighbor_distance[0] > threshold:
        return "Unknown"
    else:
        return representations[nearest_neighbor_idx[0]][0]

def get_face_enrollement_number(root_directory):
    result = []
    for file in os.listdir(root_directory):
        unknown_embedding = DeepFace.represent(img_path=os.path.join(root_directory, file), model_name="VGG-Face", enforce_detection=False)[0]["embedding"]
        classification_result = classify_image(unknown_embedding, threshold, n_neighbours=1)
        if classification_result != "Unknown":
            result.append(classification_result[-15:-4])
        else:
            result.append(classification_result)

    return result

