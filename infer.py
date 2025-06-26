#!/usr/bin/env python3

"""Infer script for the SSLZip autoencoder model."""

import argparse
import os
import re

import onnxruntime as ort
from transformers import HubertModel
import torch
import torchaudio

# Parse command line arguments.
parser = argparse.ArgumentParser()
parser.add_argument(
    "--input",
    type=str,
    required=True,
    help="Path to the input audio file. Must be 16kHz monaural.",
)
parser.add_argument(
    "--model",
    type=str,
    default="sslzip_256.onnx",
    help="Path to the autoencoder ONNX model file.",
)
parser.add_argument(
    "--output",
    type=str,
    default=None,
    help="Path to save the output latent representation.",
)
args = parser.parse_args()

# Load the upstream HuBERT model.
upstream_model = "facebook/hubert-base-ls960"
try:
    upstream = HubertModel.from_pretrained(upstream_model, local_files_only=True)
except Exception:
    upstream = HubertModel.from_pretrained(upstream_model)
upstream.eval()

# Load the autoencoder model.
postprocessor = ort.InferenceSession(args.model)
node_name = postprocessor.get_inputs()[0].name
dim = int(re.search(r"sslzip_(\d+)", args.model).group(1))

# Load the input audio file.
x, sr = torchaudio.load(args.input)
assert sr == 16000, "Input audio must have a sample rate of 16000 Hz."
assert x.shape[0] == 1, "Input audio must be monaural."

# Extract the latent representation for downstream tasks.
with torch.inference_mode():
    h = upstream(x, output_hidden_states=True).hidden_states[-1]
    assert h.shape[-1] == 768
    z = postprocessor.run(None, {node_name: h.cpu().numpy()})[0]
    assert z.shape[-1] == dim

# Save the latent representation to a file.
if args.output is None:
    output_file = os.path.splitext(args.input)[0] + ".lat"
else:
    output_file = args.output
z.tofile(output_file)

print("Done.")
