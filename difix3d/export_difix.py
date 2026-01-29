import json
from pathlib import Path
import click
import torch

import diffusers.utils.logging as diffusers_logging
import transformers.utils.logging as transformers_logging
from difix3d.model import Difix
from difix3d.convert_pipeline_to_model import convert_pipeline_to_model_state_dict
from difix3d.pipeline_difix import DifixPipeline


def export_forward_with_ref(
  model: Difix,
  model_output_dir: Path,
  input_image: torch.Tensor,
  ref_image: torch.Tensor,
  metadata: dict[str, int | str],
  model_name: str,
) -> None:
  """Export forward_with_ref variant."""
  class ForwardWithRefModule(torch.nn.Module):
    def forward(self, x, ref_image):
      return model.forward(x, ref_image=ref_image)
  
  exported_forward_with_ref = torch.export.export(ForwardWithRefModule(), (input_image, ref_image))
  torch.export.save(
    exported_forward_with_ref,
    str(model_output_dir / "forward_with_ref.pt2"),
    extra_files={"metadata.json": json.dumps(metadata)}
  )
  print(f"Exported {model_name} components saved to {model_output_dir}/")
  print("  - forward_with_ref.pt2 (forward pass with reference)")


def export_forward_no_ref(
  model: Difix,
  model_output_dir: Path,
  input_image: torch.Tensor,
  metadata: dict[str, int | str],
  model_name: str,
) -> None:
  """Export forward_no_ref variant."""
  class ForwardNoRefModule(torch.nn.Module):
    def forward(self, x):
      return model.forward(x, ref_image=None)
  
  exported_forward = torch.export.export(ForwardNoRefModule(), (input_image,))
  torch.export.save(
    exported_forward,
    str(model_output_dir / "forward.pt2"),
    extra_files={"metadata.json": json.dumps(metadata)}
  )
  print(f"Exported {model_name} components saved to {model_output_dir}/")
  print("  - forward.pt2 (forward pass without reference)")


@click.command()
@click.option("--output-path", type=str, default="output", help="Directory to save the exported model")
@click.option("--height", type=int, default=576, help="Height of the input image")
@click.option("--width", type=int, default=1024, help="Width of the input image")
@click.option("--timestep", type=int, default=199, help="Diffusion timestep")
@click.option("--model-path", type=str, default=None, help="Path to a model state dict to be used")
@click.option("--batch-size", type=int, default=1, help="Batch size for export")
@click.option("--dtype", type=click.Choice(["float32", "float16", "bfloat16"]), default="bfloat16", help="Dtype for export")
def main(output_path, height, width, timestep, model_path, batch_size, dtype):
  torch.set_grad_enabled(False)
  torch.backends.cuda.matmul.allow_tf32 = True

  # Enable progress bars for HuggingFace downloads
  diffusers_logging.enable_progress_bar()
  transformers_logging.set_verbosity_info()
  
  script_path = Path(__file__).resolve()
  checkpoint_dir = script_path.parent.parent / "checkpoints"
  checkpoint_file = checkpoint_dir / "difix.pkl"
  checkpoint_file_ref = checkpoint_dir / "difix_ref.pkl"
  
  dtype_map = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
  }
  export_dtype = dtype_map[dtype]
  
  output_dir = Path(output_path)
  output_dir.mkdir(parents=True, exist_ok=True)
  
  # Export both models
  for model_name, checkpoint_file_path, use_ref in [
    ("difix", checkpoint_file, False),
    ("difix_ref", checkpoint_file_ref, True),
  ]:
    print(f"\n{'='*60}")
    print(f"Exporting {model_name} model")
    print(f"{'='*60}")
    
    # Convert pipeline to model weights if checkpoint doesn't exist
    if not checkpoint_file_path.exists():
      print(f"Checkpoint file not found. Converting {model_name} pipeline to model weights...")
      print("  This will download: tokenizer, text_encoder (~500MB), vae (~300MB), unet (~1GB+)")
      print("  Progress bars should appear below if downloading...")
      checkpoint_dir.mkdir(parents=True, exist_ok=True)
      pretrained_name = "nvidia/difix_ref" if use_ref else "nvidia/difix"
      difix_pipeline = DifixPipeline.from_pretrained(pretrained_name, trust_remote_code=True)
      convert_pipeline_to_model_state_dict(pipeline=difix_pipeline, model_name=model_name, output_path=checkpoint_file_path)
      print("✓ Conversion completed")
    
    # Load checkpoint to check mv_unet flag
    checkpoint_sd = torch.load(checkpoint_file_path, map_location="cpu")
    checkpoint_mv_unet = checkpoint_sd.get("mv_unet")
    
    # Validate checkpoint matches expected model type
    if checkpoint_mv_unet is not None and checkpoint_mv_unet != use_ref:
      raise ValueError(f"Checkpoint mismatch: checkpoint has mv_unet={checkpoint_mv_unet} but model expects mv_unet={use_ref}")
    
    print(f"Loading {model_name} model components from HuggingFace...")
    print("  This will download: tokenizer, text_encoder (~500MB), vae (~300MB), unet (~1GB+)")
    print("  Progress bars should appear below if downloading...")
    pretrained_name = "nvidia/difix_ref" if use_ref else "nvidia/difix"
    model = Difix(
      pretrained_name=pretrained_name,
      pretrained_path=checkpoint_file_path,
      timestep=timestep,
      mv_unet=use_ref,  # Use expected mv_unet value
    )
    print("✓ Model loaded successfully")

    if model_path is not None:
      print(f"Loading custom model weights from {model_path}...")
      state_dict = torch.load(model_path, map_location="cpu")
      model.load_state_dict(state_dict)

    print("Preparing model for export...")
    model.set_eval()
    model = model.to(dtype=export_dtype)

    print(f"Exporting {model_name} components with batch size {batch_size} and dtype {dtype}...")
    model_output_dir = output_dir / model_name
    model_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Export forward variants based on model type
    with torch.autocast(device_type="cuda", dtype=export_dtype):
      input_image = torch.zeros(batch_size, 3, height, width, dtype=export_dtype).cuda()
      ref_image = torch.zeros(1, 3, height, width, dtype=export_dtype).cuda()
      
      metadata = {
        "height": height,
        "width": width,
        "dtype": dtype,
        "timestep": timestep,
        "model_name": model_name,
        "reference": use_ref,
      }
      
      if use_ref:
        export_forward_with_ref(model, model_output_dir, input_image, ref_image, metadata, model_name)
      else:
        export_forward_no_ref(model, model_output_dir, input_image, metadata, model_name)
  
  print(f"\n{'='*60}")
  print(f"All models exported successfully to {output_dir}/")
  print(f"  - {output_dir}/difix/")
  print(f"  - {output_dir}/difix_ref/")

if __name__ == "__main__":
  main()
