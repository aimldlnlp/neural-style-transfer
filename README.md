# Neural Style Transfer

This project implements Neural Style Transfer using PyTorch. It allows you to blend the content of one image with the artistic style of another image, producing visually appealing stylized images. The implementation leverages a pre-trained VGG network to extract content and style features and optimizes an output image to minimize content and style loss.

## Features

- Neural Style Transfer using VGG-based feature extraction
- Customizable style and content weights
- GPU acceleration support via PyTorch
- Simple and modular code structure
- Utilities for loading and saving images

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/aimldlnlp/neural-style-transfer.git
   cd neural-style-transfer
   ```

2. Create a Python virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

The main script `main.py` demonstrates how to perform neural style transfer:

```python
import torch
from utils.image_loader import load_image
from utils.image_saver import save_image
from nst.trainer import train_style_transfer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

content_path = "data/content1.jpg"
style_path = "data/style1.jpg"
output_path = "output/result.jpg"

content = load_image(content_path).to(device)
style = load_image(style_path, shape=content.shape[-2:]).to(device)

output = train_style_transfer(content, style, device)
save_image(output, output_path)
```

Replace the paths with your own content and style images. The output image will be saved to the specified output path.

## Dependencies

- torch
- torchvision
- Pillow

These are listed in `requirements.txt` and can be installed via pip.

## Output

The stylized images are saved in the `output/` directory. Example results are included in the repository.

## License

This project is provided as-is without any explicit license. Feel free to use and modify it for your own purposes.
