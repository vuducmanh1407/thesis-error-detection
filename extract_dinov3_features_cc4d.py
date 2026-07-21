# from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
import numpy as np
import argparse
import os
import pickle
from PIL import Image, ImageDraw, ImageFont
import cv2
import glob
from accelerate import Accelerator
from transformers import AutoImageProcessor, AutoModel
from sklearn.cluster import AgglomerativeClustering

from huggingface_hub import login
# add token here

model_id = "facebook/dinov3-vitb16-pretrain-lvd1689m"
device = "cuda" if torch.cuda.is_available() else "cpu"

processor = AutoImageProcessor.from_pretrained(model_id)
model = AutoModel.from_pretrained(model_id).to(device)

def extract_object_features(frame_list):
    obj_feature_list = list()
    framewise_feature_list = list()
    h = 960
    w = 960
    inputs = processor(images=frame_list,
                    do_resize=True,
                    size={"height": h, "width": w},
                    do_center_crop=False, # Often useful to turn off to avoid unwanted cutting
                    return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model(**inputs)
    
    pooled_output = outputs.pooler_output.detach().cpu().numpy()
    last_hidden_states = outputs.last_hidden_state

    B = last_hidden_states.shape[0]

    patch_size = model.config.patch_size # This should be 14
    # Calculate the grid dimensions (height and width of the patch feature map)
    # Note: the input image might be resized during preprocessing to the model's expected size.
    # To get the exact dimensions, we can infer from the num_patches
    feature_map_height = int(h // patch_size)
    feature_map_width = int(w // patch_size)
    for i in range(B):
        patch_tokens = last_hidden_states[i, 5:, :]
        num_patches, hidden_dim = patch_tokens.shape

        X = patch_tokens.cpu().numpy()

        cluster_model = AgglomerativeClustering(n_clusters=None, distance_threshold=75.0)
        # cluster_model = KMeans(n_clusters=20)
        # cluster_model = MeanShift(bandwidth=8.0)
        # cluster_model = DBSCAN(eps=1.0)
        clustering = cluster_model.fit_predict(X)
        output_clusters = np.reshape(clustering, (feature_map_height, feature_map_width))
        object_array = []

        # object contour finding
        for i in range(cluster_model.n_clusters_):
            binary_mask = output_clusters == i

            if binary_mask.dtype != np.uint8:
                binary_mask = (binary_mask * 255).astype(np.uint8)
            
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for i, contour in enumerate(contours):

                if cv2.contourArea(contour) >= 10:
                    # Create a blank mask for the current object
                    object_mask = np.zeros_like(binary_mask)

                    # Draw the current contour, filled in (thickness=-1)
                    cv2.drawContours(image=object_mask, contours=[contour], contourIdx=-1, color=255, thickness=-1)

                    # Get the pixel coordinates within the filled contour using np.nonzero()
                    # np.nonzero() returns a tuple of arrays: (row_indices, column_indices)
                    object_mask = object_mask > 0

                    pixels_inside = np.nonzero(object_mask)

                    # Combine row and column indices into a list of (y, x) or (row, col) coordinates
                    # Each element in pixels_list[i] is an array of coordinates for that object
                    coordinates = np.vstack((pixels_inside[0], pixels_inside[1])).T
                    obj_coord = coordinates.mean(axis=0)
                    obj_coord[0] = obj_coord[0] / feature_map_height
                    obj_coord[1] = obj_coord[1] / feature_map_width

                    # flatten
                    idx = object_mask.flatten()

                    # select and take the mean
                    obj_mean = X[idx].mean(axis=0)

                    object_array.append(np.concatenate([obj_mean, obj_coord], axis=0))

        object_array = np.vstack(object_array)
        obj_feature_list.append(object_array)
    
    return obj_feature_list, pooled_output

def process_video_frames(video_path):
    # Open the video file
    entries = os.listdir(video_path) 

    # Filter for only files
    files = [f for f in entries if os.path.isfile(os.path.join(video_path, f))]

    print(f"Starting frame extraction and processing for: {video_path}")

    frame_list = list()
    result_list = list()
    framewise_feat_list = list()

    frame_count = 0
    for filename in files:
        # Load frames
        pil_img = Image.open(os.path.join(video_path, filename)).convert('RGB')
        frame_list.append(pil_img)

        if len(frame_list) == 20:
            obj_feature_list, pooled_output = extract_object_features(frame_list)
            result_list = result_list + obj_feature_list
            framewise_feat_list.append(pooled_output)
            frame_list = list()

        # Example: Display information about the frame
        if frame_count % 100 == 0: # Print a message every 100 frames
            print(f"Processing frame {frame_count}: Size {pil_img.size}, Mode {pil_img.mode}")
        # ----------------------------------------------

        # Read the next frame
        frame_count += 1

    if len(frame_list) > 0:
        obj_feature_list, pooled_output = extract_object_features(frame_list)
        result_list = result_list + obj_feature_list
        framewise_feat_list.append(pooled_output)

    print(f"Finished processing {frame_count} frames.")
    return result_list, np.concatenate(framewise_feat_list, axis=0)


parser = argparse.ArgumentParser(description="Dataset")
parser.add_argument('--dataset', '-d', type=str, required=True, help='')
args = parser.parse_args()

dataset = args.dataset

video_dir = os.path.join("./", dataset, "frames_10fps")
save_dir = os.path.join("./", dataset, "dinov3_obj_feats")
frame_feat_dir = os.path.join("./", dataset, "frame_features_dinov3")
os.makedirs(save_dir, exist_ok=True)
os.makedirs(frame_feat_dir, exist_ok=True)

def list_folder_names_os(directory_path):
    """
    Lists the names of all folders in the specified directory.
    """
    folder_names = []
    # Iterate over all entries in the directory
    for entry in os.listdir(directory_path):
        # Construct the full path to check if it's a directory
        full_path = os.path.join(directory_path, entry)
        # Check if the entry is a directory
        if os.path.isdir(full_path):
            folder_names.append(entry)
    return folder_names

file_names = list_folder_names_os(video_dir)

for name in file_names:
    save_path = os.path.join(save_dir, name + ".pkl")
    save_feat_path = os.path.join(frame_feat_dir, name + ".npy")

    # Check if a video is being processed
    if os.path.isfile(save_path): # and os.path.isfile(save_feat_path):
        print(f"The {name} file exists. Skipping.")
        continue
    
    print(f"Processing {name}.")
    # Create a dummy file
    with open(save_path, 'wb') as f:
        pickle.dump([], f)

    # Processing
    file_path = os.path.join(video_dir, name)
    results, framewise_feats = process_video_frames(file_path)

    # Save
    with open(os.path.join(save_path), 'wb') as f:
        pickle.dump(results, f)

    np.save(save_feat_path, framewise_feats, allow_pickle=True)