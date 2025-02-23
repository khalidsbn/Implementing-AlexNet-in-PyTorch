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