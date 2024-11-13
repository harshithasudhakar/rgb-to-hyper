# test_generator.py

import tensorflow as tf
from model import Generator

def test_generator():
    # Instantiate the Generator with the correct parameter
    generator = Generator(HSI_CHANNELS=31)
    
    # Build the model with the expected input shape
    generator.build(input_shape=(None, 256, 256, 3))
    
    # Print the model summary to verify architecture
    generator.summary()
    
    # Create a dummy input tensor
    dummy_input = tf.random.normal([4, 256, 256, 31])
    
    try:
        # Perform a forward pass
        generated_output = generator(dummy_input, training=False)
        print(f"Generated HSI shape: {generated_output.shape}")
    except Exception as e:
        print(f"Error during generation: {e}")

if __name__ == "__main__":
    test_generator()
