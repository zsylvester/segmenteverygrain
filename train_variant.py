import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import segmenteverygrain as seg

input_dir = "noisy_output/"
patch_dir = "patches/"
image_dir, mask_dir = seg.patchify_training_data(input_dir, patch_dir)

train_dataset, val_dataset, test_dataset = seg.create_train_val_test_data(
    image_dir, mask_dir, augmentation=True
)

model_name = "testsyntheticmodifiedep100lr2"
model = seg.create_and_train_model_from_pretrained(
    "models/seg_model.keras",
    train_dataset,
    val_dataset,
    test_dataset,
    epochs=100,
    learning_rate=1e-2,
    model_type="unet_modified",
    save_plot_path=f"loss_plots/training_loss_plot{model_name}.png",
    show_plot=False,
    use_reduce_lr=True
)

model.save(f"models/{model_name}.keras")
