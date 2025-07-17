#!/usr/bin/env python3
"""
Script to convert both difix and difix_ref pipeline models to Difix model state dicts.
"""

import torch
from difix3d.pipeline_difix import DifixPipeline
from convert_pipeline_to_model import (
  convert_pipeline_to_model_state_dict,
  convert_pipeline_to_full_model_state_dict,
)


def main():
  print("=== Converting DifixPipeline Models ===\n")

  # Convert difix model
  print("1. Converting difix model...")
  try:
    difix_pipeline = DifixPipeline.from_pretrained("nvidia/difix")

    # Convert to minimal state dict
    minimal_state_dict = convert_pipeline_to_model_state_dict(
      pipeline=difix_pipeline, output_path="model.pkl"
    )
    print("✓ Difix minimal conversion successful")

    # Convert to full state dict
    full_state_dict = convert_pipeline_to_full_model_state_dict(
      pipeline=difix_pipeline, output_path="full_model.pkl"
    )
    print("✓ Difix full conversion successful")

  except Exception as e:
    print(f"✗ Error converting difix model: {e}")

  # Convert difix_ref model
  print("\n2. Converting difix_ref model...")
  try:
    difix_ref_pipeline = DifixPipeline.from_pretrained("nvidia/difix_ref")

    # Convert to minimal state dict
    minimal_state_dict = convert_pipeline_to_model_state_dict(
      pipeline=difix_ref_pipeline, output_path="model.pkl"
    )
    print("✓ Difix_ref minimal conversion successful")

    # Convert to full state dict
    full_state_dict = convert_pipeline_to_full_model_state_dict(
      pipeline=difix_ref_pipeline, output_path="full_model.pkl"
    )
    print("✓ Difix_ref full conversion successful")

  except Exception as e:
    print(f"✗ Error converting difix_ref model: {e}")

  print("\n=== Conversion completed! ===")
  print("Generated files:")
  print("- models/difix/model.pkl (minimal)")
  print("- models/difix/full_model.pkl (complete)")
  print("- models/difix_ref/model.pkl (minimal)")
  print("- models/difix_ref/full_model.pkl (complete)")


if __name__ == "__main__":
  main()
