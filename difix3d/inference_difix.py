import os
import argparse
import time
from glob import glob
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from difix3d.model import Difix
import torch
import torch.nn.functional as F


def main():
  # Argument parser
  parser = argparse.ArgumentParser()
  parser.add_argument("input_image", type=str, help="Path to the input image or directory")
  parser.add_argument(
    "--ref_image", type=str, default=None, help="Path to the reference image or directory"
  )

  parser.add_argument("--height", type=int, default=576, help="Height of the input image")
  parser.add_argument("--width", type=int, default=1024, help="Width of the input image")

  parser.add_argument(
    "--model_path", type=str, default=None, help="Path to a model state dict to be used"
  )
  parser.add_argument(
    "--output_dir", type=str, default="output", help="Directory to save the output"
  )
  parser.add_argument("--seed", type=int, default=42, help="Random seed to be used")
  parser.add_argument("--timestep", type=int, default=199, help="Diffusion timestep")
  parser.add_argument("--num_iterations", type=int, default=100, help="Number of iterations for benchmarking")
  args = parser.parse_args()

  # Create output directory
  os.makedirs(args.output_dir, exist_ok=True)

  # Initialize the model
  has_ref = args.ref_image is not None
  checkpoint_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "checkpoints")
  model = Difix(
    pretrained_name="nvidia/difix_ref" if has_ref else "nvidia/difix",
    pretrained_path=os.path.join(checkpoint_dir, "difix_ref.pkl" if has_ref else "difix.pkl"),
    timestep=args.timestep,
    mv_unet=has_ref,
  )

  torch.set_grad_enabled(False)

  model.set_eval()
  
  # Convert model parameters to bfloat16 before export
  model = model.to(dtype=torch.bfloat16)
  x = torch.zeros(1 if args.ref_image is None else 2, 3, args.height, args.width).cuda()
  
  with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    exported_program = torch.export.export(model, (x,))

  model = exported_program.module()
  model = torch.compile(model, backend="torch_tensorrt", dynamic=False, options={"truncate_long_and_double": True, "enabled_precisions": {torch.float16}})


  # Load single image for benchmarking
  T = transforms.Compose([
    transforms.ToTensor(),  # Converts PIL to tensor in [0, 1] range
  ])
  
  pil_image = Image.open(args.input_image).convert("RGB")
  image = T(pil_image).unsqueeze(0).cuda().to(torch.bfloat16)  # (1, c, h, w)
  
  ref_image_tensor = None
  if args.ref_image is not None:
    pil_ref = Image.open(args.ref_image).convert("RGB")
    ref_image_tensor = T(pil_ref).unsqueeze(0).cuda().to(torch.bfloat16)  # (1, c, h, w)

  # Warmup
  print("Warming up...")
  for _ in range(10):
    _ = sample_image(model, image, height=args.height, width=args.width, ref_image=ref_image_tensor)
  
  # Benchmark
  print(f"Benchmarking {args.num_iterations} iterations...")
  torch.cuda.synchronize()
  start_time = time.time()
  
  for _ in tqdm(range(args.num_iterations), desc="Benchmarking"):
    output_image = sample_image(model, image, height=args.height, width=args.width, ref_image=ref_image_tensor)
  
  torch.cuda.synchronize()
  end_time = time.time()
  
  total_time = end_time - start_time
  avg_time = total_time / args.num_iterations
  fps = args.num_iterations / total_time
  
  print(f"\nBenchmark Results:")
  print(f"  Total time: {total_time:.3f}s")
  print(f"  Average time per iteration: {avg_time*1000:.2f}ms")
  print(f"  Throughput: {fps:.2f} FPS")


def sample_image(model, image, width, height, ref_image=None):
  """Sample an image using either a regular model or exported model.
  
  Args:
    model: Model (regular or exported) to use for inference
    image: Input tensor of shape (b, c, h, w) already on device, in [0, 1] range
    width: Target width
    height: Target height
    ref_image: Optional reference tensor of shape (b, c, h, w) already on device, in [0, 1] range
    
  Returns:
    Output tensor of shape (b, c, h, w) on device, normalized to [-1, 1]
  """
  # Normalize from [0, 1] to [-1, 1]
  image = image * 2.0 - 1.0
  
  # Resize using interpolate
  image = F.interpolate(image, size=(height, width), mode="bilinear", align_corners=False)
  
  if ref_image is None:
    x = image  # (b, c, h, w)
  else:
    # Normalize and resize ref_image
    ref_image = ref_image * 2.0 - 1.0
    ref_image = F.interpolate(ref_image, size=(height, width), mode="bilinear", align_corners=False)
    x = torch.cat([image, ref_image], dim=0)  # (2*b, c, h, w)

  with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    output_image = model(x)  # Returns (b, c, h, w)
  return output_image

  

if __name__ == "__main__":
  main()
