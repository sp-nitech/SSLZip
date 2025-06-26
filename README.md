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

| Upstream Model | Dimensions | w/ CLUB | ONNX Model |
| -------------- | ---------- | ------- | ---------- |
| HuBERT Base    | 256        | ✔️       | [link](https://huggingface.co) |
| HuBERT Base    | 256        |         | [link](https://huggingface.co) |
| HuBERT Base    | 16         |         | [link](https://huggingface.co) |

## Citation

```bibtex
@InProceedings{sp-nitech2023sptk,
  author = {Takenori Yoshimura and Shinji Takaki and Kazuhiro Nakamura and Keiichiro Oura and Takato Fujimoto and Kei Hashimoto and Yoshihiko Nankaku and Keiichi Tokuda},
  title = {{SSLZip}:Simple autoencoding for enhancing self-supervised speech representations in speech generation},
  booktitle = {13th ISCA Speech Synthesis Workshop (SSW 2025)},
  pages = {xxx--xxx},
  year = {2025},
}
```
