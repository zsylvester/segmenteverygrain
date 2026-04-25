"""
Edge completion CNN training for image segmentation.
Trains a CNN to predict edge regions instead of using mirroring.
"""

import os
import numpy as np
from glob import glob
from PIL import Image
import tensorflow as tf
from keras import Model
from keras.layers import Input, Conv2D, Conv2DTranspose, concatenate, LeakyReLU, BatchNormalization
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import argparse


I = 256
I2 = 128
EDGE_WIDTH = 32


def load_images_from_directory(image_dir, extensions=['png', 'jpg', 'jpeg', 'tif', 'tiff']):
    """
    Load all images from a directory.
    
    Parameters
    ----------
    image_dir : str
        Path to directory containing images.
    extensions : list
        List of valid image extensions.
    
    Returns
    -------
    images : list
        List of numpy arrays (images).
    """
    images = []
    for ext in extensions:
        patterns = [os.path.join(image_dir, f'*.{ext}'), 
                  os.path.join(image_dir, f'*.{ext.upper()}')]
        for pattern in patterns:
            files = glob(pattern)
            for f in files:
                if '_mask' in os.path.basename(f).lower():
                    continue
                img = np.array(Image.open(f).convert('RGB'))
                images.append(img)
    return images


def prepare_edge_training_data(images, edge_width=EDGE_WIDTH):
    """
    Prepare training data for edge completion CNN.
    Input: inner region (no edges)
    Output: edge regions
    
    Parameters
    ----------
    images : list
        List of full image arrays.
    edge_width : int
        Width of edge bands to extract.
    
    Returns
    -------
    inner_list : list
        Inner regions (inputs).
    edge_lists : list
        [top_list, bottom_list, left_list, right_list] (targets).
    """
    inner_list = []
    top_list = []
    bottom_list = []
    left_list = []
    right_list = []
    
    for img in images:
        h, w = img.shape[:2]
        
        if h < 2*edge_width or w < 2*edge_width:
            continue
        
        inner = img[edge_width:h-edge_width, edge_width:w-edge_width]
        top = img[:edge_width, edge_width:w-edge_width]
        bottom = img[h-edge_width:, edge_width:w-edge_width]
        left = img[edge_width:h-edge_width, :edge_width]
        right = img[edge_width:h-edge_width, w-edge_width:]
        
        inner_list.append(inner)
        top_list.append(top)
        bottom_list.append(bottom)
        left_list.append(left)
        right_list.append(right)
    
    print(f"Prepared {len(inner_list)} training pairs")
    
    return inner_list, [top_list, bottom_list, left_list, right_list]


def save_training_data(X, edge_lists, output_dir):
    """
    Save training data to disk.
    
    Parameters
    ----------
    X : list
        Input patches.
    edge_lists : list
        List of [top_list, bottom_list, left_list, right_list].
    output_dir : str
        Directory to save data.
    """
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, 'X_inner.npy'), np.array(X, dtype=object))
    top, bottom, left, right = edge_lists
    np.save(os.path.join(output_dir, 'y_top.npy'), np.array(top, dtype=object))
    np.save(os.path.join(output_dir, 'y_bottom.npy'), np.array(bottom, dtype=object))
    np.save(os.path.join(output_dir, 'y_left.npy'), np.array(left, dtype=object))
    np.save(os.path.join(output_dir, 'y_right.npy'), np.array(right, dtype=object))
    print(f"Saved {len(X)} inputs to {output_dir}")


def load_training_data(input_dir):
    """
    Load training data from disk.
    
    Parameters
    ----------
    input_dir : str
        Directory containing saved training data.
    
    Returns
    -------
    X : list
        Inner regions.
    edge_lists : list
        [top_list, bottom_list, left_list, right_list].
    """
    X = np.load(os.path.join(input_dir, 'X_inner.npy'), allow_pickle=True).tolist()
    top = np.load(os.path.join(input_dir, 'y_top.npy'), allow_pickle=True).tolist()
    bottom = np.load(os.path.join(input_dir, 'y_bottom.npy'), allow_pickle=True).tolist()
    left = np.load(os.path.join(input_dir, 'y_left.npy'), allow_pickle=True).tolist()
    right = np.load(os.path.join(input_dir, 'y_right.npy'), allow_pickle=True).tolist()
    return X, [top, bottom, left, right]


def create_edge_completion_model(input_shape=(None, None, 3)):
    """
    Create CNN for edge completion.
    Input: inner region
    Output: same size (identity) - just applies learned refinement
    
    Parameters
    ----------
    input_shape : tuple
        Shape of input images.
    
    Returns
    -------
    model : keras.Model
        The model.
    """
    inputs = Input(shape=input_shape)
    
    x = Conv2D(32, (3, 3), padding='same')(inputs)
    x = LeakyReLU(0.2)(x)
    x = BatchNormalization()(x)
    
    x = Conv2D(32, (3, 3), padding='same')(x)
    x = LeakyReLU(0.2)(x)
    x = BatchNormalization()(x)
    
    outputs = Conv2D(3, (1, 1), activation='sigmoid')(x)
    
    model = Model(inputs, outputs)
    return model


def predict_edges(inner, edge_width=EDGE_WIDTH, model=None):
    """
    Predict edge completion for an inner region.
    
    Parameters
    ----------
    inner : numpy.ndarray
        Inner region (H x W x 3), normalized to [0, 1].
    edge_width : int
        Width of edges to predict.
    model : keras.Model, dict, or None
        Trained model(s). If dict with keys top/bottom/left/right, uses those.
        If None, uses mirroring.
    
    Returns
    -------
    top, bottom, left, right : numpy.ndarray
        Predicted edge strips.
    """
    if model is None:
        top = np.flip(inner[:edge_width, :], axis=0)
        bottom = np.flip(inner[-edge_width:, :], axis=0)
        left = np.flip(inner[:, :edge_width], axis=1)
        right = np.flip(inner[:, -edge_width:], axis=1)
        return top, bottom, left, right
    
    if isinstance(model, dict):
        model_top = model['top']
        model_bottom = model['bottom']
        model_left = model['left']
        model_right = model['right']
        
        top_input = inner[:edge_width, :, :]
        bottom_input = inner[-edge_width:, :, :]
        left_input = inner[:, :edge_width, :]
        right_input = inner[:, -edge_width:, :]
        
        top_batch = np.expand_dims(top_input, axis=0)
        bottom_batch = np.expand_dims(bottom_input, axis=0)
        left_batch = np.expand_dims(left_input, axis=0)
        right_batch = np.expand_dims(right_input, axis=0)
        
        top = model_top.predict(top_batch, verbose=0)[0]
        bottom = model_bottom.predict(bottom_batch, verbose=0)[0]
        left = model_left.predict(left_batch, verbose=0)[0]
        right = model_right.predict(right_batch, verbose=0)[0]
        
        return top, bottom, left, right
    
    inner_batch = np.expand_dims(inner, axis=0)
    pred = model.predict(inner_batch, verbose=0)[0]
    
    h, w = inner.shape[:2]
    top = pred[:edge_width, :w]
    bottom = pred[h-edge_width:, :w]
    left = pred[:, :edge_width]
    right = pred[:, w-edge_width:]
    
    return top, bottom, left, right

def train_edge_completion_model(image_dir, output_dir, epochs=50, learning_rate=1e-3, edge_width=EDGE_WIDTH):
    """
    Train the edge completion model.
    Train 4 separate models for top, bottom, left, right edges.
    
    Parameters
    ----------
    image_dir : str
        Directory with training images.
    output_dir : str
        Directory to save model.
    epochs : int
        Training epochs.
    learning_rate : float
        Learning rate.
    edge_width : int
        Edge width.
        
    Returns
    -------
    model : keras.Model
        Trained model for top edges.
    """
    print("Loading images...")
    images = load_images_from_directory(image_dir)
    print(f"Loaded {len(images)} images")
    
    print("Preparing training data...")
    inner_list, edge_lists = prepare_edge_training_data(images, edge_width=edge_width)
    top_list, bottom_list, left_list, right_list = edge_lists
    
    X = np.array(inner_list, dtype=np.float32) / 255.0
    y_top = np.array(top_list, dtype=np.float32) / 255.0
    y_bottom = np.array(bottom_list, dtype=np.float32) / 255.0
    y_left = np.array(left_list, dtype=np.float32) / 255.0
    y_right = np.array(right_list, dtype=np.float32) / 255.0
    
    print(f"X (inner) shape: {X.shape}")
    print(f"y_top shape: {y_top.shape}")
    print(f"y_bottom shape: {y_bottom.shape}")
    print(f"y_left shape: {y_left.shape}")
    print(f"y_right shape: {y_right.shape}")
    
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    split = int(0.8 * len(indices))
    train_idx, val_idx = indices[:split], indices[split:]
    
    def create_edge_model(name):
        inputs = Input(shape=(None, None, 3), name=name)
        
        x = Conv2D(32, (3, 3), padding='same')(inputs)
        x = LeakyReLU(0.2)(x)
        x = BatchNormalization()(x)
        
        x = Conv2D(64, (3, 3), padding='same')(x)
        x = LeakyReLU(0.2)(x)
        x = BatchNormalization()(x)
        
        x = Conv2D(128, (3, 3), padding='same')(x)
        x = LeakyReLU(0.2)(x)
        x = BatchNormalization()(x)
        
        x = Conv2D(64, (3, 3), padding='same')(x)
        x = LeakyReLU(0.2)(x)
        x = BatchNormalization()(x)
        
        x = Conv2D(32, (3, 3), padding='same')(x)
        x = LeakyReLU(0.2)(x)
        
        outputs = Conv2D(3, (1, 1), activation='sigmoid')(x)
        model = Model(inputs, outputs)
        return model
    
    os.makedirs(output_dir, exist_ok=True)
    
    model_top = create_edge_model('top')
    model_top.compile(optimizer=Adam(learning_rate), loss='mse')
    print(f"\nTraining top edge model...")
    model_top.fit(
        X[train_idx, :edge_width, :, :], y_top[train_idx],
        validation_data=(X[val_idx, :edge_width, :, :], y_top[val_idx]),
        epochs=epochs, batch_size=4, verbose=1
    )
    model_top.save(os.path.join(output_dir, 'edge_model_top.keras'))
    print(f"Saved top model")
    
    model_bottom = create_edge_model('bottom')
    model_bottom.compile(optimizer=Adam(learning_rate), loss='mse')
    print(f"\nTraining bottom edge model...")
    model_bottom.fit(
        X[train_idx, -edge_width:, :, :], y_bottom[train_idx],
        validation_data=(X[val_idx, -edge_width:, :, :], y_bottom[val_idx]),
        epochs=epochs, batch_size=4, verbose=1
    )
    model_bottom.save(os.path.join(output_dir, 'edge_model_bottom.keras'))
    print(f"Saved bottom model")
    
    model_left = create_edge_model('left')
    model_left.compile(optimizer=Adam(learning_rate), loss='mse')
    print(f"\nTraining left edge model...")
    model_left.fit(
        X[train_idx, :, :edge_width, :], y_left[train_idx],
        validation_data=(X[val_idx, :, :edge_width, :], y_left[val_idx]),
        epochs=epochs, batch_size=4, verbose=1
    )
    model_left.save(os.path.join(output_dir, 'edge_model_left.keras'))
    print(f"Saved left model")
    
    model_right = create_edge_model('right')
    model_right.compile(optimizer=Adam(learning_rate), loss='mse')
    print(f"\nTraining right edge model...")
    model_right.fit(
        X[train_idx, :, -edge_width:, :], y_right[train_idx],
        validation_data=(X[val_idx, :, -edge_width:, :], y_right[val_idx]),
        epochs=epochs, batch_size=4, verbose=1
    )
    model_right.save(os.path.join(output_dir, 'edge_model_right.keras'))
    print(f"Saved right model")
    
    print(f"\nAll models saved to {output_dir}")
    
    return model_top


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, required=True, help='Directory with input images')
    parser.add_argument('--output_dir', type=str, default='edge_training_data/', help='Output directory')
    parser.add_argument('--edge_width', type=int, default=EDGE_WIDTH, help='Edge width')
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    args = parser.parse_args()
    
    train_edge_completion_model(
        args.image_dir, 
        args.output_dir, 
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        edge_width=args.edge_width
    )