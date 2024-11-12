# rgb-to-hyper
end-to-end implementation of microplastics detection in water using hyperspectral imaging.

# How it works

Summary of the Training and Prediction Flow

A. Training Flow (mode = "global"):

1. Model and Optimizer Initialization:
   - Instantiate Generator and Discriminator.
   - Initialize their respective Adam optimizers.

2. Logging Setup:
   - Create a summary_writer for TensorBoard to log training metrics.

3. Checkpoint Path Determination:
     - Set checkpoint_path to 'global_ckpt' for global training.

4. Training Execution:
   - Call train_gan with all necessary parameters.
   - Inside train_gan:
     - Checkpoint Restoration: Load existing checkpoints if available.
     - Data Loading and Preparation: Load and preprocess paired RGB and HSI images.
     - Epoch Loop: For each epoch, shuffle data and iterate over batches.
       - Batch Processing: Perform augmentation, train discriminator, train generator, compute metrics, and log progress.
       - Checkpoint Saving: Save model states at the end of each epoch.
     - Post-Training: Save final metrics and optionally generate sample outputs.


B. Prediction Flow (mode = "predict")

1. Checkpoint Restoration:
  - Use load_model_and_predict to load the Generator model from the 'global_ckpt' directory.

2. Data Loading:
   - Load RGB images designated for prediction.

3. HSI Generation:
  - Use the restored Generator to create HSI images from the RGB inputs.

4. Output Saving:
   - Save the generated HSI images as TIFF files in the specified directory.
