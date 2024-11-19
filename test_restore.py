# test_restore.py

import tensorflow as tf
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def list_all_checkpoints(checkpoint_dir):
    """
    Lists all available checkpoints in the specified directory.
    Args:
        checkpoint_dir (str): Path to the checkpoints directory.
    Returns:
        None
    """
    checkpoints = tf.train.get_checkpoint_state(checkpoint_dir)
           
    if checkpoints and checkpoints.all_model_checkpoint_paths:
        logging.info("Available checkpoints:")
        for path in checkpoints.all_model_checkpoint_paths:
            print(path)
    else:
        logging.error("No checkpoints found in the specified directory.")
if __name__ == "__main__":
    CHECKPOINT_DIR = r'C:\Harshi\ecs-venv\rgb-to-hyper\rgb-to-hyper-main\rgb-to-hyper\checkpoints'
    list_all_checkpoints(CHECKPOINT_DIR)
