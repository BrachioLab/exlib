# exlib
[![PyPI](https://img.shields.io/pypi/v/exlib)](https://pypi.org/project/exlib/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/BrachioLab/exlib/blob/master/LICENSE)

`exlib` is a comprehensive package showcasing our lab's work on explanation methods, featuring user-friendly modules for easy application of various techniques. 

## Installation
```
pip install exlib
```

<!--
If you have exlib already installed, please check that you have the latest version:
```
python -c "import exlib; print(exlib.__version__)"
# This should print "0.1.0". If it does not, update the package by running:
pip install -U exlib
```
-->

To use `pytorch-gradcam`, install our customized and expanded version at
```
pip install grad-cam@git+https://github.com/brachiolab/pytorch-grad-cam
```

## Projects
We list below some relevant projects that use exlib heavily.

### The FIX Benchmark: Extracting Features Interpretable to eXperts
* Documentation available [here](https://github.com/BrachioLab/exlib/tree/main/fix).
* Quick-start tutorial notebook at [`fix_demo.py`](https://colab.research.google.com/github/BrachioLab/exlib/blob/main/fix/fix_demo.ipynb)
* [<a href="https://arxiv.org/abs/2409.13684">Paper</a>] [<a href="https://brachiolab.github.io/fix/">Website</a>] [<a href="https://debugml.github.io/fix/">Blog Post</a>]

## New Explainers (src/exlib/new_explainers/)
Low-abstraction implementations of popular feature attribution methods with **segment-level explanations** by default.

### Example: LimeImage with ResNet50
```python
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from exlib.new_explainers import LimeImage

# Load model
model = models.resnet50(weights='DEFAULT').eval()  # or weights=None for random init

# Load the cute buccee image
image_pil = Image.open('tests/buccee.jpg').convert('RGB')
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
image = transform(image_pil)

# Create LIME explainer and generate explanation
explainer = LimeImage(n_samples=1000, patch_size=16)
explanation = explainer.explain(model, image)   # Auto targets the top class

print(f"Attribution shape: {explanation.attributions.shape}")  # (3, 224, 224)
print(f"Segments shape: {explanation.segments.shape}")        # (3, 224, 224)
print(f"Number of segments: {explanation.metadata['num_segments']}")  # 196
print(f"R² score: {explanation.r2_score:.3f}")               # Model fit quality
```

### Example: ShapText with GPT-2
```python
import torch
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification
from exlib.new_explainers import ShapText

# Load model and tokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2ForSequenceClassification.from_pretrained('gpt2').to(device).eval()
model.config.pad_token_id = 0

# Tokenize text
text = "The quick brown fox jumps"
input_ids = tokenizer.encode(text, return_tensors='pt')[0].to(device)

# Create SHAP explainer and generate explanation
explainer = ShapText(embedding_layer=model.transformer.wte, n_samples=1000)
explanation = explainer.explain(model, input_ids)

print(f"Token attributions: {explanation.attributions}")      # Shape: (num_tokens,)
print(f"Expected value: {explanation.expected_value:.3f}")    # Baseline prediction
for token, attr in zip(tokenizer.convert_ids_to_tokens(input_ids), explanation.attributions):
    print(f"  {token}: {attr:.3f}")
```


