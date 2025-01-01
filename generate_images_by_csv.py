import argparse
import os
from tqdm import tqdm
import pandas as pd
import torch
from RegionalDiffusion_base import RegionalDiffusionPipeline
from RegionalDiffusion_xl import RegionalDiffusionXLPipeline
from diffusers.schedulers import DPMSolverMultistepScheduler

# Initialize the pipeline
pipe = RegionalDiffusionXLPipeline.from_single_file(
    "https://huggingface.co/svjack/GenshinImpact_XL_Base/blob/main/sdxlBase_v10.safetensors",
    torch_dtype=torch.float16, use_safetensors=True, variant="fp16"
)
pipe.to("cuda")
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)
pipe.enable_xformers_memory_efficient_attention()

# Negative prompt
negative_prompt = ""

def generate_images(data_df, output_dir, times, base_ratio, num_inference_steps, height, width, guidance_scale):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Iterate over each row in the DataFrame
    for _, row in tqdm(data_df.iterrows(), total=len(data_df), desc="Processing rows"):
        action = row["action"]
        prompt = row["prompt"]
        split_ratio = row["split_ratio"]
        regional_prompt = row["regional_prompt"]

        # Generate `times` images for each row
        for i in range(times):
            # Generate the image
            try:
                image = pipe(
                    prompt=regional_prompt,
                    split_ratio=split_ratio,
                    batch_size=1,
                    base_ratio=base_ratio,
                    base_prompt=prompt,
                    num_inference_steps=num_inference_steps,
                    height=height,
                    negative_prompt=negative_prompt,
                    width=width,
                    seed=i,  # Use the iteration index as the seed for variability
                    guidance_scale=guidance_scale
                ).images[0]

                # Save the image
                image_filename = f"{action}_{i+1}.png"  # e.g., "action_1.png", "action_2.png", etc.
                image_path = os.path.join(output_dir, image_filename)
                image.save(image_path)
            except:
                print("err action: {}".format(action))
                continue

def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Generate images from a CSV file using RegionalDiffusionXLPipeline.")
    parser.add_argument("--input_csv", type=str, required=True, help="Path to the input CSV file.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the generated images.")
    parser.add_argument("--times", type=int, default=5, help="Number of times to repeat the generation for each row.")
    parser.add_argument("--base_ratio", type=float, default=0.5, help="The ratio of the base prompt.")
    parser.add_argument("--num_inference_steps", type=int, default=20, help="Number of sampling steps.")
    parser.add_argument("--height", type=int, default=1024, help="Height of the generated image.")
    parser.add_argument("--width", type=int, default=1024, help="Width of the generated image.")
    parser.add_argument("--guidance_scale", type=float, default=7.0, help="Guidance scale for the generation.")
    args = parser.parse_args()

    # Load the CSV file
    data_df = pd.read_csv(args.input_csv)

    # Generate and save images
    generate_images(
        data_df,
        args.output_dir,
        args.times,
        args.base_ratio,
        args.num_inference_steps,
        args.height,
        args.width,
        args.guidance_scale
    )

if __name__ == "__main__":
    main()