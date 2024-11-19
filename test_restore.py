# Testing if Checkpoint is getting saved properly, need to update with corruption check

import os
import logging
import tensorflow as tf
from config import CHECKPOINT_DIR
from model import Generator, Discriminator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("test_restore.log"),
        logging.StreamHandler()
    ]
)

def test_restore():
    generator = Generator()
    discriminator = Discriminator()

    checkpoint_path = os.path.join(CHECKPOINT_DIR, 'global_ckpt')

    # Initialize Checkpoint Without Optimizers
    checkpoint = tf.train.Checkpoint(
        generator=generator,
        discriminator=discriminator
    )

    checkpoint_manager = tf.train.CheckpointManager(
        checkpoint,
        directory=checkpoint_path,
        max_to_keep=5
    )

    if checkpoint_manager.latest_checkpoint:
        checkpoint.restore(checkpoint_manager.latest_checkpoint).expect_partial()
        logging.info(f"Successfully restored from checkpoint: {checkpoint_manager.latest_checkpoint}")
    else:
        logging.info("No checkpoint found. Initializing from scratch.")

if __name__ == "__main__":
    test_restore()
