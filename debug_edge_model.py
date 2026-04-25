import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import os
import sys

sys.path.insert(0, '/Users/aedancook-shen/Desktop/FineGrainBenchmarking/segmenteverygrain')
from train_edge_completion import predict_edges

model_dir = '/Users/aedancook-shen/Desktop/FineGrainBenchmarking/segmenteverygrain/models'
EDGE_WIDTH = 32

model_top = keras.models.load_model(os.path.join(model_dir, 'edge_model_top.keras'))
model_bottom = keras.models.load_model(os.path.join(model_dir, 'edge_model_bottom.keras'))
model_left = keras.models.load_model(os.path.join(model_dir, 'edge_model_left.keras'))
model_right = keras.models.load_model(os.path.join(model_dir, 'edge_model_right.keras'))

test_images_dir = '/Users/aedancook-shen/Desktop/FineGrainBenchmarking/segmenteverygrain/testcleanimages'
image_files = [f for f in os.listdir(test_images_dir) if f.endswith('.tif') or f.endswith('.tiff')]
image_path = os.path.join(test_images_dir, image_files[0])
image = Image.open(image_path).convert('RGB')
image = np.array(image).astype(np.float32) / 255.0

print(f"Image shape: {image.shape}")

h, w = image.shape[:2]
inner = image

result_mirror = predict_edges(inner, edge_width=EDGE_WIDTH, model=None)
result_model = predict_edges(inner, edge_width=EDGE_WIDTH, model={
    'top': model_top, 'bottom': model_bottom, 'left': model_left, 'right': model_right
})

top_mirror, bottom_mirror, left_mirror, right_mirror = result_mirror
top_model, bottom_model, left_model, right_model = result_model

diff_top = np.abs(top_mirror.astype(float) - top_model.astype(float))
print(f"Top edge - Max diff: {diff_top.max():.3f}, Mean diff: {diff_top.mean():.3f}")
print(f"Top edge - mirror mean: {top_mirror.mean():.3f}, model mean: {top_model.mean():.3f}")
print(f"Top edge - mirror std: {top_mirror.std():.3f}, model std: {top_model.std():.3f}")

diff_bottom = np.abs(bottom_mirror.astype(float) - bottom_model.astype(float))
print(f"Bottom edge - Max diff: {diff_bottom.max():.3f}, Mean diff: {diff_bottom.mean():.3f}")

diff_left = np.abs(left_mirror.astype(float) - left_model.astype(float))
print(f"Left edge - Max diff: {diff_left.max():.3f}, Mean diff: {diff_left.mean():.3f}")

diff_right = np.abs(right_mirror.astype(float) - right_model.astype(float))
print(f"Right edge - Max diff: {diff_right.max():.3f}, Mean diff: {diff_right.mean():.3f}")