import json
from pathlib import Path
import click
import torch

import diffusers.utils.logging as diffusers_logging
import transformers.utils.logging as transformers_logging
from difix3d.model import Difix
from difix3d.convert_pipeline_to_model import convert_pipeline_to_model_state_dict
from difix3d.pipeline_difix import DifixPipeline
from difix3d.convert_pipeline_to_model import convert_pipeline_to_model_state_dict
from difix3d.pipeline_difix import DifixPipeline


@click.command()
@click.option("--output-path", type=str, default="output", help="Directory to save the exported model")
@click.option("--height", type=int, default=576, help="Height of the input image")
@click.option("--width", type=int, default=1024, help="Width of the input image")
@click.option("--timestep", type=int, default=199, help="Diffusion timestep")
@click.option("--model-path", type=str, default=None, help="Path to a model state dict to be used")
@click.option("--batch-size", type=int, default=1, help="Batch size for export")
@click.option("--dtype", type=click.Choice(["float32", "float16", "bfloat16"]), default="bfloat16", help="Dtype for export")
@click.option("--reference", is_flag=True, default=False, help="Export model with reference image support (batch_size=2)")
def main(output_path, height, width, timestep, model_path, batch_size, dtype, reference):
  torch.set_grad_enabled(False)
  torch.backends.cuda.matmul.allow_tf32 = True

  # Enable progress bars for HuggingFace downloads
  diffusers_logging.enable_progress_bar()
  transformers_logging.set_verbosity_info()
  
  script_path = Path(__file__).resolve()
  checkpoint_dir = script_path.parent.parent / "checkpoints"
  checkpoint_file = checkpoint_dir / "difix.pkl"
  
  # Convert pipeline to model weights if checkpoint doesn't exist
  if not checkpoint_file.exists():
    print("Checkpoint file not found. Converting pipeline to model weights...")
    print("  This will download: tokenizer, text_encoder (~500MB), vae (~300MB), unet (~1GB+)")
    print("  Progress bars should appear below if downloading...")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    difix_pipeline = DifixPipeline.from_pretrained("nvidia/difix", trust_remote_code=True)
    convert_pipeline_to_model_state_dict(pipeline=difix_pipeline, model_name="difix", output_path=checkpoint_file)
    print("✓ Conversion completed")
  
  print("Loading Difix model components from HuggingFace...")
  print("  This will download: tokenizer, text_encoder (~500MB), vae (~300MB), unet (~1GB+)")
  print("  Progress bars should appear below if downloading...")
  model = Difix(
    pretrained_name="nvidia/difix",
    pretrained_path=checkpoint_file,
    timestep=timestep,
    mv_unet=False,
  )
  print("✓ Model loaded successfully")

  if model_path is not None:
    print(f"Loading custom model weights from {model_path}...")
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)

  print("Preparing model for export...")
  model.set_eval()
  
  dtype_map = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
  }
  export_dtype = dtype_map[dtype]
  model = model.to(dtype=export_dtype)

  print(f"Exporting model with batch size {batch_size} and dtype {dtype}...")
  output_dir = Path(output_path)
  output_dir.mkdir(parents=True, exist_ok=True)
  
  # Export both versions: with and without reference
  for ref_mode in [False, True]:
    # Reference mode uses batch_size=2 (two 3-channel images stacked), not 6 channels
    export_batch_size = 2 if ref_mode else batch_size
    inputs = torch.zeros(export_batch_size, 3, height, width, dtype=export_dtype).cuda()
    
    with torch.autocast(device_type="cuda", dtype=export_dtype):
      exported_program = torch.export.export(model, (inputs,))

    print(f"Saving exported model ({'with' if ref_mode else 'without'} reference)...")
    model_file = output_dir / ("difix3d_ref.pt2" if ref_mode else "difix3d.pt2")
    
    metadata = {
      "height": height,
      "width": width,
      "dtype": dtype,
      "reference": ref_mode,
    }
    
    torch.export.save(exported_program, str(model_file), extra_files={"metadata.json": json.dumps(metadata)})
    print(f"Exported model saved to {model_file}")

if __name__ == "__main__":
  main()
