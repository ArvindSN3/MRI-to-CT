from dataclasses import dataclass
import torch
from diffusers import DDPMPipeline
from diffusers.utils import make_image_grid
import os

@dataclass
class TrainingConfig:
    image_size = 256  # the generated image resolution
    train_batch_size = 4
    eval_batch_size = 16  # how many images to sample during evaluation
    num_epochs = 30
    gradient_accumulation_steps = 1
    learning_rate = 1e-4
    lr_warmup_steps = 500
    save_image_epochs = 5
    save_model_epochs = 10
    mixed_precision = "fp16"  # `no` for float32, `fp16` for automatic mixed precision
    output_dir = "ct-trained-sd-1-5"  # the model name locally and on the HF Hub

    push_to_hub = True  # whether to upload the saved model to the HF Hub
    hub_model_id = "Crimson-Dawn/ct-trained-sd-1-5"  # the name of the repository to create on the HF Hub
    hub_private_repo = True
    overwrite_output_dir = True  # overwrite the old model when re-running the notebook
    seed = 0


config = TrainingConfig()



def evaluate(config, epoch, pipeline):
    # Sample some images from random noise (this is the backward diffusion process).
    # The default pipeline output type is `List[PIL.Image]`
    images = pipeline(
        batch_size=config.eval_batch_size,
        generator=torch.manual_seed(config.seed),
    ).images

    # Make a grid out of the images
    image_grid = make_image_grid(images, rows=4, cols=4)

    # Save the images
    test_dir = os.path.join(config.output_dir, "samples")
    os.makedirs(test_dir, exist_ok=True)
    image_grid.save(f"{test_dir}/{epoch:04d}.png")
    

model_id = "Crimson-Dawn/ct-trained-sd-1-5"
pipeline = DDPMPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
evaluate(config, config.num_epochs, pipeline)