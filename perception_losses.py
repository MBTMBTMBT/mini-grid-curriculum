import torch
from torchvision.models import vgg11, vgg13, vgg16, vgg19
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152

# VGG Models: Extracting shallow, mid, and deep layers
# VGG11: Shallow (first two convolution blocks), Mid (three blocks), Deep (all blocks)
vgg11_layers_shallow = vgg11(pretrained=True).features[:8]   # First 2 conv blocks
vgg11_layers_mid = vgg11(pretrained=True).features[:13]      # First 3 conv blocks
vgg11_layers_deep = vgg11(pretrained=True).features[:]       # All conv blocks

# VGG13: Shallow (first two convolution blocks), Mid (three blocks), Deep (all blocks)
vgg13_layers_shallow = vgg13(pretrained=True).features[:8]
vgg13_layers_mid = vgg13(pretrained=True).features[:13]
vgg13_layers_deep = vgg13(pretrained=True).features[:]

# VGG16: Shallow (first two convolution blocks), Mid (three blocks), Deep (all blocks)
vgg16_layers_shallow = vgg16(pretrained=True).features[:8]
vgg16_layers_mid = vgg16(pretrained=True).features[:16]
vgg16_layers_deep = vgg16(pretrained=True).features[:]

# VGG19: Shallow (first two convolution blocks), Mid (three blocks), Deep (all blocks)
vgg19_layers_shallow = vgg19(pretrained=True).features[:8]
vgg19_layers_mid = vgg19(pretrained=True).features[:16]
vgg19_layers_deep = vgg19(pretrained=True).features[:]

# ResNet Models: Extracting layers from early to late stages
# ResNet18: Shallow (up to layer2), Mid (up to layer3), Deep (all layers)
resnet18_layers_shallow = torch.nn.Sequential(*list(resnet18(pretrained=True).children())[:6])  # Up to layer2
resnet18_layers_mid = torch.nn.Sequential(*list(resnet18(pretrained=True).children())[:7])      # Up to layer3
resnet18_layers_deep = torch.nn.Sequential(*list(resnet18(pretrained=True).children())[:8])     # All layers

# ResNet34: Same structure, different depth in each layer
resnet34_layers_shallow = torch.nn.Sequential(*list(resnet34(pretrained=True).children())[:6])  # Up to layer2
resnet34_layers_mid = torch.nn.Sequential(*list(resnet34(pretrained=True).children())[:7])      # Up to layer3
resnet34_layers_deep = torch.nn.Sequential(*list(resnet34(pretrained=True).children())[:8])     # All layers

# ResNet50: Shallow (up to layer2), Mid (up to layer3), Deep (all layers)
resnet50_layers_shallow = torch.nn.Sequential(*list(resnet50(pretrained=True).children())[:6])  # Up to layer2
resnet50_layers_mid = torch.nn.Sequential(*list(resnet50(pretrained=True).children())[:7])      # Up to layer3
resnet50_layers_deep = torch.nn.Sequential(*list(resnet50(pretrained=True).children())[:8])     # All layers

# ResNet101: Shallow (up to layer2), Mid (up to layer3), Deep (all layers)
resnet101_layers_shallow = torch.nn.Sequential(*list(resnet101(pretrained=True).children())[:6])  # Up to layer2
resnet101_layers_mid = torch.nn.Sequential(*list(resnet101(pretrained=True).children())[:7])      # Up to layer3
resnet101_layers_deep = torch.nn.Sequential(*list(resnet101(pretrained=True).children())[:8])     # All layers

# ResNet152: Shallow (up to layer2), Mid (up to layer3), Deep (all layers)
resnet152_layers_shallow = torch.nn.Sequential(*list(resnet152(pretrained=True).children())[:6])  # Up to layer2
resnet152_layers_mid = torch.nn.Sequential(*list(resnet152(pretrained=True).children())[:7])      # Up to layer3
resnet152_layers_deep = torch.nn.Sequential(*list(resnet152(pretrained=True).children())[:8])     # All layers
