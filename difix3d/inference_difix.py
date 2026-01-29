import time
import json
import pickle
from pathlib import Path
from tqdm import tqdm
import cv2
import numpy as np
import click
import torch
import torch.nn.functional as F
import torch_tensorrt

DataType = {
  torch.float32: "float32",
  torch.float16: "float16",
  torch.bfloat16: "bfloat16",
  torch.int8: "int8",
}

DtypeMap = {v: k for k, v in DataType.items()}

def load_image_tensor(image_path: str, target_height: int, target_width: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
  cv_image = cv2.imread(image_path)
  assert cv_image is not None, f"Failed to load image: {image_path}"
  # Resize on CPU before moving to GPU to avoid OOM
  cv_image = cv2.resize(cv_image, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
  image = torch.from_numpy(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float() / 255.0
  image = (image - 0.5) / 0.5
  return image.unsqueeze(0).cuda().to(dtype=dtype).contiguous()

def tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
  normalized = (tensor.cpu().float() + 1.0) / 2.0
  clamped = normalized.clamp(0, 1)
  single_image = clamped.squeeze(0)
  hwc = single_image.permute(1, 2, 0).numpy()
  return (hwc * 255).astype(np.uint8)

def compile_tensorrt(exported_program, example_input, exported_model_path: Path, workspace_size_gb: int, enabled_precisions, dtype: torch.dtype, verbose: bool):
  """Compile exported program with TensorRT, using cache if available."""
  capability = torch.cuda.get_device_capability()
  sm_version = f"sm{capability[0]}{capability[1]}"
  
  precision_str = "_".join(sorted(DataType.get(prec, str(prec)) for prec in enabled_precisions))
  cache_dir = exported_model_path.parent
  cache_file = cache_dir / f"{exported_model_path.stem}_trt_{sm_version}_{precision_str}.pt2"
  
  if cache_file.exists():
    print(f"Loading cached TensorRT model from {cache_file}...")
    return torch.load(cache_file, map_location="cuda", weights_only=False)
  
  print(f"Compiling TensorRT model (will cache to {cache_file})...")
  workspace_size_bytes = workspace_size_gb * 1024 * 1024 * 1024
  compile_kwargs = {"workspace_size": workspace_size_bytes, "enabled_precisions": enabled_precisions}
  
  with torch.autocast(device_type="cuda", dtype=dtype):
    if verbose:
      with torch_tensorrt.logging.info():
        compiled_model = torch_tensorrt.dynamo.compile(exported_program, ([example_input],), **compile_kwargs)
    else:
      compiled_model = torch_tensorrt.dynamo.compile(exported_program, ([example_input],), **compile_kwargs)
  
  print(f"Saving TensorRT model to {cache_file}...")
  torch.save(compiled_model, cache_file, pickle_protocol=pickle.HIGHEST_PROTOCOL)
  return compiled_model

def sample_image(model, image, dtype: torch.dtype, reference: torch.Tensor | None = None):
  """Sample an image using the exported model.
  
  Args:
    model: Exported model to use for inference
    image: Input tensor of shape (b, c, h, w) already on device, in [-1, 1] range
    dtype: Dtype to use for autocast
    reference: Optional reference tensor of shape (b, c, h, w), required if model uses reference
    
  Returns:
    Output tensor of shape (b, c, h, w) on device, normalized to [-1, 1]
  """
  if reference is not None:
    # Stack along batch dimension (batch_size=2) instead of channel dimension
    inputs = torch.cat([image, reference], dim=0)
  else:
    inputs = image
  with torch.autocast(device_type="cuda", dtype=dtype):
    output_image = model(inputs)
  # If reference was used, model returns batch_size=2, take first element
  if reference is not None:
    output_image = output_image[0:1]
  return output_image

def load_model(exported_model: str, tensorrt: bool, compile_flag: bool, workspace_size: int, verbose: bool, max_autotune: bool = False):
  """Load and compile the exported model.
  
  Returns:
    Tuple of (compiled_model, height, width, dtype, reference)
  """
  extra_files = {"metadata.json": ""}
  exported_program = torch.export.load(exported_model, extra_files=extra_files)
  metadata = json.loads(extra_files["metadata.json"])
  
  height = metadata["height"]
  width = metadata["width"]
  reference = metadata.get("reference", False)
  
  # Always use dtype from metadata (model was exported with this dtype)
  dtype = DtypeMap[metadata["dtype"]]
  
  # Reference mode uses batch_size=2 (two 3-channel images stacked), not 6 channels
  batch_size = 2 if reference else 1
  x = torch.zeros(batch_size, 3, height, width, dtype=dtype).cuda()
  exported_path = Path(exported_model)
  
  if tensorrt:
    enabled_precisions = {dtype}
    compiled_model = compile_tensorrt(exported_program, x, exported_path, workspace_size, enabled_precisions, dtype, verbose)
  elif compile_flag:
    model_module = exported_program.module()
    # Ensure model is in correct dtype before compilation
    model_module = model_module.to(dtype=dtype)
    if max_autotune:
      compiled_model = torch.compile(model_module, mode="max-autotune")
    else:
      compiled_model = torch.compile(model_module, fullgraph=True)
  else:
    compiled_model = exported_program.module()
  
  return compiled_model, height, width, dtype, reference

def run_benchmark(compiled_model, image: torch.Tensor, num_iterations: int, dtype: torch.dtype, reference: torch.Tensor | None = None):
  """Run benchmark on the compiled model."""
  print("Warming up...")
  for _ in range(10):
    _ = sample_image(compiled_model, image, dtype, reference)
  
  print(f"Benchmarking {num_iterations} iterations...")
  torch.cuda.synchronize()
  start_time = time.time()
  
  for _ in tqdm(range(num_iterations), desc="Benchmarking"):
    output_image = sample_image(compiled_model, image, dtype, reference)
  
  torch.cuda.synchronize()
  end_time = time.time()
  
  total_time = end_time - start_time
  avg_time = total_time / num_iterations
  fps = num_iterations / total_time
  
  print("\nBenchmark Results:")
  print(f"  Total time: {total_time:.3f}s")
  print(f"  Average time per iteration: {avg_time*1000:.2f}ms")
  print(f"  Throughput: {fps:.2f} FPS")

def run_inference_display(compiled_model, image: torch.Tensor, input_image: str, height: int, width: int, dtype: torch.dtype, reference: torch.Tensor | None = None, reference_image: str | None = None):
  """Run inference and display results."""
  print("Warming up...")
  for _ in range(10):
    _ = sample_image(compiled_model, image, dtype, reference)
  
  print("Running inference...")
  output_image = sample_image(compiled_model, image, dtype, reference)
  output_np = tensor_to_image(output_image)
  
  # Load original input image for display
  input_cv = cv2.imread(input_image)
  input_resized = cv2.resize(input_cv, (width, height), interpolation=cv2.INTER_LINEAR)
  
  # Convert output to BGR for cv2
  output_bgr = cv2.cvtColor(output_np, cv2.COLOR_RGB2BGR)
  
  # Concatenate side by side
  side_by_side = np.hstack([input_resized, output_bgr])
  
  cv2.imshow("Input | Output", side_by_side)
  print("Press any key to close...")
  cv2.waitKey(0)
  cv2.destroyAllWindows()

@click.command()
@click.option("--exported-model", type=str, default=None, help="Path to the exported model file")
@click.argument("input_image", type=str)
@click.option("--benchmark", is_flag=True, help="Run benchmark instead of showing images")
@click.option("--num-iterations", type=int, default=100, help="Number of iterations for benchmarking")
@click.option("--tensorrt", is_flag=True, help="Use TensorRT compilation")
@click.option("--compile", is_flag=True, help="Use torch.compile")
@click.option("--workspace-size", type=int, default=20, help="TensorRT workspace size in GB (default: 20GB)")
@click.option("--verbose", is_flag=True, help="Enable verbose TensorRT output")
@click.option("--max-autotune", is_flag=True, help="Use max-autotune mode for torch.compile")
@click.option("--ref", type=str, default=None, help="Reference image path")
def main(exported_model, input_image, benchmark, num_iterations, tensorrt, compile, workspace_size, verbose, max_autotune, ref):
  torch.set_grad_enabled(False)
  torch.backends.cuda.matmul.allow_tf32 = True
  
  if exported_model is None:
    exported_model = "output/difix3d_ref.pt2" if ref is not None else "output/difix3d.pt2"
  
  compiled_model, height, width, dtype, uses_reference = load_model(exported_model, tensorrt, compile, workspace_size, verbose, max_autotune)
  
  if (ref is not None) != uses_reference:
    if uses_reference:
      raise ValueError(f"Model at {exported_model} requires a reference image (use --ref)")
    else:
      raise ValueError(f"Model at {exported_model} does not support reference images (remove --ref)")
  
  # Preprocess and load image on device before benchmark
  image = load_image_tensor(input_image, height, width, dtype=torch.float32)
  
  ref_tensor = None
  if uses_reference:
    ref_tensor = load_image_tensor(ref, height, width, dtype=torch.float32)
  
  # Ensure image is ready for inference
  torch.cuda.synchronize()

  if benchmark:
    run_benchmark(compiled_model, image, num_iterations, dtype, ref_tensor)
  else:
    run_inference_display(compiled_model, image, input_image, height, width, dtype, ref_tensor, ref)

if __name__ == "__main__":
  main()
