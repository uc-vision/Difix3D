import json
from pathlib import Path
import click
import torch

import diffusers.utils.logging as diffusers_logging
import transformers.utils.logging as transformers_logging
from difix3d.model import Difix


@click.command()
@click.option("--output-path", type=str, default="output", help="Directory to save the exported model")
@click.option("--height", type=int, default=576, help="Height of the input image")
@click.option("--width", type=int, default=1024, help="Width of the input image")
@click.option("--timestep", type=int, default=199, help="Diffusion timestep")
@click.option("--model-path", type=str, default=None, help="Path to a model state dict to be used")
@click.option("--batch-size", type=int, default=1, help="Batch size for export")
def main(output_path, height, width, timestep, model_path, batch_size):
  torch.set_grad_enabled(False)
  torch.backends.cuda.matmul.allow_tf32 = True

  # Enable progress bars for HuggingFace downloads
  diffusers_logging.enable_progress_bar()
  transformers_logging.set_verbosity_info()
  
  print("Loading Difix model components from HuggingFace...")
  print("  This will download: tokenizer, text_encoder (~500MB), vae (~300MB), unet (~1GB+)")
  print("  Progress bars should appear below if downloading...")
  script_path = Path(__file__).resolve()
  checkpoint_dir = script_path.parent.parent / "checkpoints"
  model = Difix(
    pretrained_name="nvidia/difix",
    pretrained_path=str(checkpoint_dir / "difix.pkl"),
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
  model = model.to(dtype=torch.bfloat16)

  print(f"Exporting model with batch size {batch_size}...")
  x = torch.zeros(batch_size, 3, height, width).cuda()
  
  with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    exported_program = torch.export.export(model, (x,))

  print("Saving exported model...")
  output_dir = Path(output_path)
  output_dir.mkdir(parents=True, exist_ok=True)
  model_file = output_dir / "difix3d.pt2"
  
  metadata = {
    "height": height,
    "width": width,
    "timestep": timestep,
    "batch_size": batch_size,
  }
  
  torch.export.save(exported_program, str(model_file), extra_files={"metadata.json": json.dumps(metadata)})

  print(f"Exported model saved to {model_file}")

if __name__ == "__main__":
  main()
