from diffusers import StableDiffusionPipeline
import torch
from pathlib import Path

# Load model only once (important)
MODEL_ID = "runwayml/stable-diffusion-v1-5"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

pipe = StableDiffusionPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
)
pipe = pipe.to(device)


def generate_building_images(building_type: str, floors: int, area: float, count: int = 2):
    images = []

    for i in range(count):
        prompt = (
            f"A realistic modern {building_type} building, "
            f"{floors} floors, suitable for {area} square meters land, "
            f"architectural exterior design, variation {i+1}, "
            f"photorealistic, daylight"
        )

        image = pipe(prompt, guidance_scale=7.5, num_inference_steps=30).images[0]

        path = Path("generated_images") / f"{building_type}_{i+1}.png"
        image.save(path)
        images.append(str(path))

    return images
