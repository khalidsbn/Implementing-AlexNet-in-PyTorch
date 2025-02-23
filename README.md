<div align="center">
    <h1> Implementing-AlexNet-in-PyTorch</h1>
</div>

# Implementation code in PyTorch
```
import torch  
import torch.nn as nn  

class AlexNet(nn.Module):  
    def __init__(self, num_classes=1000):  
        super(AlexNet, self).__init__()  
        self.features = nn.Sequential(  
            # Conv1: Input 224x224x3 → Output 54x54x96  
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0),  
            nn.ReLU(inplace=True),  
            nn.MaxPool2d(kernel_size=3, stride=2),  # Output 26x26x96  
            nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2),  

            # Conv2: Output 26x26x256  
            nn.Conv2d(96, 256, kernel_size=5, padding=2, groups=2),  
            nn.ReLU(inplace=True),  
            nn.MaxPool2d(kernel_size=3, stride=2),  # Output 12x12x256  
            nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2),  

            # Conv3-5: Maintain spatial resolution  
            nn.Conv2d(256, 384, kernel_size=3, padding=1),  
            nn.ReLU(inplace=True),  
            nn.Conv2d(384, 384, kernel_size=3, padding=1, groups=2),  
            nn.ReLU(inplace=True),  
            nn.Conv2d(384, 256, kernel_size=3, padding=1, groups=2),  
            nn.ReLU(inplace=True),  
            nn.MaxPool2d(kernel_size=3, stride=2),  # Output 5x5x256  
        )  
        self.classifier = nn.Sequential(  
            nn.Dropout(p=0.5),  
            nn.Linear(256 * 5 * 5, 4096),  
            nn.ReLU(inplace=True),  
            nn.Dropout(p=0.5),  
            nn.Linear(4096, 4096),  
            nn.ReLU(inplace=True),  
            nn.Linear(4096, num_classes),  
        )  

    def forward(self, x):  
        x = self.features(x)  
        x = torch.flatten(x, 1)  
        x = self.classifier(x)  
        return x  
```

# Breaking down the code: 
At its core, AlexNet is composed of eight learned layers:
* **Five convolutional layers** extract hierarchical features from the input images.
* **Three fully connected layers** that perform the final classification into 1000 ImageNet categories

The network takes an input image of size **224x224x3** (width, height, channels) and processes it through a series of convolutions, pooling, and normalization layers before flattening the features into a vector for classification. 

## Layer-by-layer Breakdown
### 1. First Convolutional Block
* **Convolution:** The first layer applies 96 kernels of size 11x11 with a stride of 4.  
  Output size = `(224 - 11) / 4 + 1 = 54`  
  So, the output is approximately **54x54x96**.
* **Max Pooling**: A **3x3** max pooling with a stride of 2 reduces the feature maps to about **26x26x96**.
* **Local Response Normalization (LRN)**: Normalization helps in generalization and was used in the original paper. 
