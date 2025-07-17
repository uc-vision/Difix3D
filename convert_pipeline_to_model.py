import torch
import pickle
import os
from difix3d.pipeline_difix import DifixPipeline
from difix3d.model import Difix


def convert_pipeline_to_model_state_dict(pipeline, model_name):
  """
  Convert a DifixPipeline to a state dict that can be loaded into a Difix model.

  Args:
      pipeline: DifixPipeline instance
      model_name: Model name for the output file
  """

  # Create output directory
  output_dir = "checkpoints"
  os.makedirs(output_dir, exist_ok=True)

  # Set output path
  output_path = os.path.join(output_dir, f"{model_name}.pkl")

  # Extract state dicts from pipeline components
  state_dict = {}

  # Get VAE LoRA configuration from the pipeline
  state_dict["vae_lora_target_modules"] = list(pipeline.vae.peft_config["vae_skip"].target_modules)
  state_dict["rank_vae"] = pipeline.vae.peft_config["vae_skip"].r

  state_dict["state_dict_unet"] = pipeline.unet.state_dict()

  # VAE state dict - filter for LoRA and skip parameters like in Difix model
  vae_state_dict = pipeline.vae.state_dict()
  state_dict["state_dict_vae"] = {
    k: v for k, v in vae_state_dict.items() if "lora" in k or "skip" in k
  }

  # Create a dummy optimizer state dict (required by Difix model)
  # We'll create a temporary model just for the optimizer
  temp_model = Difix()
  optimizer = torch.optim.AdamW(temp_model.parameters(), lr=1e-4)
  state_dict["optimizer"] = optimizer.state_dict()

  # Save to pickle file
  print(f"Saving converted state dict to {output_path}")
  torch.save(state_dict, output_path)

  print(f"State dict converted and saved successfully to {output_path}")
  print(f"State dict keys: {list(state_dict.keys())}")
  print(f"UNet state dict keys: {list(state_dict['state_dict_unet'].keys())}")
  print(f"VAE state dict keys: {list(state_dict['state_dict_vae'].keys())}")

  return state_dict


def load_converted_state_dict(pickle_path, model_path=None):
  """
  Load converted state dict from pickle file into a Difix model.

  Args:
      pickle_path: Path to the pickle file containing converted state dict
      model_path: Path to pretrained Difix checkpoint (optional)

  Returns:
      Difix model with loaded state dict
  """

  # Load state dict from pickle
  print(f"Loading converted state dict from {pickle_path}")
  state_dict = torch.load(pickle_path, map_location="cpu")

  # Initialize Difix model
  model = Difix()

  # Use the same loading logic as the original model
  from difix3d.model import load_ckpt_from_state_dict

  optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
  model, _ = load_ckpt_from_state_dict(model, optimizer, pickle_path)

  print("Converted state dict loaded successfully into Difix model")
  return model


if __name__ == "__main__":
  # Convert difix model
  print("Converting difix model...")
  difix_pipeline = DifixPipeline.from_pretrained("nvidia/difix", trust_remote_code=True)
  convert_pipeline_to_model_state_dict(pipeline=difix_pipeline, model_name="difix")

  # Convert difix_ref model
  print("\nConverting difix_ref model...")
  difix_ref_pipeline = DifixPipeline.from_pretrained("nvidia/difix_ref", trust_remote_code=True)
  convert_pipeline_to_model_state_dict(pipeline=difix_ref_pipeline, model_name="difix_ref")

  print("\nConversion completed!")
  print("Generated files:")
  print("- checkpoints/difix.pkl")
  print("- checkpoints/difix_ref.pkl")
