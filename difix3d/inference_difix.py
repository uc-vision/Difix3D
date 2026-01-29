import time
import json
from dataclasses import dataclass
from pathlib import Path
from tqdm import tqdm
import cv2
import numpy as np
import click
import torch

DataType = {
  torch.float32: "float32",
  torch.float16: "float16",
  torch.bfloat16: "bfloat16",
  torch.int8: "int8",
}

DtypeMap = {v: k for k, v in DataType.items()}

@dataclass
class ExportedModels:
  forward_model: torch.nn.Module
  forward_with_ref_model: torch.nn.Module
  height: int
  width: int
  dtype: torch.dtype

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
  single_image = clamped.squeeze(0)  # Batch size is always 1
  hwc = single_image.permute(1, 2, 0).numpy()
  return (hwc * 255).astype(np.uint8)

def load_and_compile_model(model_path: Path, compile_flag: bool, max_autotune: bool):
  """Load and optionally compile a single exported model.
  
  Args:
    model_path: Path to the .pt2 file
    compile_flag: Whether to use torch.compile
    max_autotune: Whether to use max-autotune mode for torch.compile
    
  Returns:
    Compiled model module
  """
  program = torch.export.load(str(model_path))
  model = program.module()
  if compile_flag:
    if max_autotune:
      model = torch.compile(model, mode="max-autotune")
    else:
      model = torch.compile(model, fullgraph=True)
  return model

def load_models(export_dir: str, use_reference: bool, compile_flag: bool, max_autotune: bool = False) -> ExportedModels:
  """Load the exported model components.
  
  Args:
    export_dir: Base directory containing difix/ and difix_ref/ subdirectories
    use_reference: Whether to load the reference-enabled model (difix_ref) or standard model (difix)
    compile_flag: Whether to use torch.compile
    max_autotune: Whether to use max-autotune mode for torch.compile
    
  Returns:
    ExportedModels dataclass containing all models and metadata
  """
  export_path = Path(export_dir)
  model_name = "difix_ref" if use_reference else "difix"
  model_path = export_path / model_name
  
  # Load metadata and forward model based on model type
  if use_reference:
    # difix_ref: only has forward_with_ref
    forward_file = model_path / "forward_with_ref.pt2"
    extra_files = {"metadata.json": ""}
    torch.export.load(str(forward_file), extra_files=extra_files)
    metadata = json.loads(extra_files["metadata.json"])
    forward_model = None  # difix_ref doesn't have forward without ref
    forward_with_ref_model = load_and_compile_model(forward_file, compile_flag, max_autotune)
  else:
    # difix: only has forward
    forward_file = model_path / "forward.pt2"
    extra_files = {"metadata.json": ""}
    torch.export.load(str(forward_file), extra_files=extra_files)
    metadata = json.loads(extra_files["metadata.json"])
    forward_model = load_and_compile_model(forward_file, compile_flag, max_autotune)
    forward_with_ref_model = None  # difix doesn't have forward with ref
  
  height = metadata["height"]
  width = metadata["width"]
  dtype = DtypeMap[metadata["dtype"]]
  
  return ExportedModels(forward_model, forward_with_ref_model, height, width, dtype)

def sample_image(models: ExportedModels, image: torch.Tensor, reference: torch.Tensor | None = None):
  """Sample an image using the exported model components.
  
  Args:
    models: ExportedModels dataclass containing all models and metadata
    image: Input tensor of shape (1, c, h, w) already on device, in [-1, 1] range
    reference: Optional reference tensor of shape (1, c, h, w)
    
  Returns:
    Output tensor of shape (1, c, h, w) on device, normalized to [-1, 1]
  """
  with torch.autocast(device_type="cuda", dtype=models.dtype):
    if reference is not None:
      if models.forward_with_ref_model is None:
        raise ValueError("Reference image provided but model doesn't support reference (use difix_ref model)")
      output_image = models.forward_with_ref_model(image, reference)
    else:
      if models.forward_model is None:
        raise ValueError("No reference image provided but model requires reference (use difix model or provide --ref)")
      output_image = models.forward_model(image)
  
  return output_image

def run_benchmark(models: ExportedModels, image: torch.Tensor, num_iterations: int, reference: torch.Tensor | None = None):
  """Run benchmark on the compiled models."""
  print("Warming up...")
  for _ in range(10):
    _ = sample_image(models, image, reference)
  
  print(f"Benchmarking {num_iterations} iterations...")
  torch.cuda.synchronize()
  start_time = time.time()
  
  for _ in tqdm(range(num_iterations), desc="Benchmarking"):
    output_image = sample_image(models, image, reference)
  
  torch.cuda.synchronize()
  end_time = time.time()
  
  total_time = end_time - start_time
  avg_time = total_time / num_iterations
  fps = num_iterations / total_time
  
  print("\nBenchmark Results:")
  print(f"  Total time: {total_time:.3f}s")
  print(f"  Average time per iteration: {avg_time*1000:.2f}ms")
  print(f"  Throughput: {fps:.2f} FPS")

def run_inference_display(models: ExportedModels, image: torch.Tensor, input_image: str, reference: torch.Tensor | None = None, reference_image: str | None = None):
  """Run inference and display results."""
  print("Warming up...")
  for _ in range(10):
    _ = sample_image(models, image, reference)
  
  print("Running inference...")
  output_image = sample_image(models, image, reference)
  output_np = tensor_to_image(output_image)
  
  # Load original input image for display
  input_cv = cv2.imread(input_image)
  input_resized = cv2.resize(input_cv, (models.width, models.height), interpolation=cv2.INTER_LINEAR)
  
  # Convert output to BGR for cv2
  output_bgr = cv2.cvtColor(output_np, cv2.COLOR_RGB2BGR)
  
  # Concatenate side by side
  side_by_side = np.hstack([input_resized, output_bgr])
  
  cv2.imshow("Input | Output", side_by_side)
  print("Press any key to close...")
  cv2.waitKey(0)
  cv2.destroyAllWindows()

@click.command()
@click.option("--export-dir", type=str, default="output", help="Directory containing exported model files")
@click.argument("input_image", type=str)
@click.option("--benchmark", is_flag=True, help="Run benchmark instead of showing images")
@click.option("--num-iterations", type=int, default=100, help="Number of iterations for benchmarking")
@click.option("--compile", is_flag=True, help="Use torch.compile")
@click.option("--max-autotune", is_flag=True, help="Use max-autotune mode for torch.compile")
@click.option("--ref", type=str, default=None, help="Reference image path")
def main(export_dir, input_image, benchmark, num_iterations, compile, max_autotune, ref):
  torch.set_grad_enabled(False)
  torch.backends.cuda.matmul.allow_tf32 = True
  
  use_reference = ref is not None
  models = load_models(export_dir, use_reference, compile, max_autotune)
  
  # Preprocess and load image on device before benchmark
  image = load_image_tensor(input_image, models.height, models.width, dtype=torch.float32)
  
  ref_tensor = None
  if ref is not None:
    ref_tensor = load_image_tensor(ref, models.height, models.width, dtype=torch.float32)
  
  # Ensure image is ready for inference
  torch.cuda.synchronize()

  if benchmark:
    run_benchmark(models, image, num_iterations, ref_tensor)
  else:
    run_inference_display(models, image, input_image, ref_tensor, ref)

if __name__ == "__main__":
  main()
