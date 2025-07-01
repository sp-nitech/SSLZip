# SSLZip

This repository provides a simple example of inference using the pretrained models of SSLZip.

## Example

```py
import onnxruntime as ort
from transformers import HubertModel
import torch

# Load the upstream HuBERT model.
upstream = HubertModel.from_pretrained("facebook/hubert-base-ls960")
upstream.eval()

# Load the autoencoder model.
postprocessor = ort.InferenceSession("sslzip_256.onnx")
node_name = postprocessor.get_inputs()[0].name

# Prepare an input waveform (assuming 16kHz audio).
x = torch.randn(1, 16000)

# Extract the latent representation for downstream tasks.
with torch.inference_mode():
    h = upstream(x, output_hidden_states=True).hidden_states[-1]
    z = postprocessor.run(None, {node_name: h.cpu().numpy()})[0]

# Use z as you like.
print(z.shape)
```

## Pretrained Models

The pretrained models were developed using the LibriSpeech corpus and are distributed under the same license (CC BY 4.0).  
Please include credit to Nagoya Institue of Technology and Techno-Speech, Inc. when using these models.

| Upstream Model | Dimensions | w/ CLUB | ONNX Model |
| -------------- | ---------- | ------- | ---------- |
| HuBERT Base    | 256        | ✓       | [link](https://huggingface.co/takenori-y/SSLZip-256-CLUB/resolve/main/sslzip_256_club.onnx) |
| HuBERT Base    | 256        |         | [link](https://huggingface.co/takenori-y/SSLZip-256/resolve/main/sslzip_256.onnx)           |
| HuBERT Base    | 16         |         | [link](https://huggingface.co/takenori-y/SSLZip-16/resolve/main/sslzip_16.onnx)             |

## Citation

```bibtex
@InProceedings{yoshimura2025sslzip,
  author = {Takenori Yoshimura and Shinji Takaki and Kazuhiro Nakamura and Keiichiro Oura and Takato Fujimoto and Kei Hashimoto and Yoshihiko Nankaku and Keiichi Tokuda},
  title = {{SSLZip}: Simple autoencoding for enhancing self-supervised speech representations in speech generation},
  booktitle = {13th ISCA Speech Synthesis Workshop (SSW 2025)},
  pages = {xxx--xxx},
  year = {2025},
}
```
