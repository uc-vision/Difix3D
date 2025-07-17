import argparse
import os
from .pipeline_difix import DifixPipeline
from diffusers.utils.loading_utils import load_image


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("input_image", type=str)
  parser.add_argument("--ref_image", type=str)
  parser.add_argument("--prompt", type=str, default="remove degradation")
  args = parser.parse_args()

  pipe = DifixPipeline.from_pretrained(
    "nvidia/difix_ref" if args.ref_image is not None else "nvidia/difix", trust_remote_code=True
  )

  pipe.to("cuda")

  input_image = load_image(args.input_image)
  ref_image = None

  if args.ref_image is not None:
    ref_image = load_image(args.ref_image)
  output_image = pipe(
    args.prompt,
    image=input_image,
    ref_image=ref_image,
    num_inference_steps=1,
    timesteps=[199],
    guidance_scale=0.0,
  ).images[0]

  output_image.save(os.path.join("output", os.path.basename(args.input_image)))
