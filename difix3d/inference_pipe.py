import argparse
import os
from .pipeline_difix import DifixPipeline
from diffusers.utils.loading_utils import load_image
import torch
from tqdm import tqdm


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("input_image", type=str)
  parser.add_argument("--ref_image", type=str)
  parser.add_argument("--prompt", type=str, default="remove degradation")
  args = parser.parse_args()

  torch.set_grad_enabled(False)

  pipe = DifixPipeline.from_pretrained(
    "nvidia/difix_ref" if args.ref_image is not None else "nvidia/difix", trust_remote_code=True,
   torch_dtype=torch.float16
  )

  pipe.to("cuda")

  input_image = load_image(args.input_image)
  ref_image = None

  if args.ref_image is not None:
    ref_image = load_image(args.ref_image)
    ref_image = pipe.image_processor.preprocess(ref_image)

  input_image = pipe.image_processor.preprocess(input_image)


  with torch.autocast("cuda", torch.float16):

    output_image = pipe(
      args.prompt,
      image=input_image,
      ref_image=ref_image,
      num_inference_steps=1,
      timesteps=[199],
      guidance_scale=0.0,
    ).images[0]

  output_image.save(os.path.join("output", os.path.basename(args.input_image)))
