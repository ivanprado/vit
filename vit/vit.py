import timm
import torch

m = timm.create_model('vit_base_patch16_224', pretrained=True)
m.eval()

o = m(torch.randn(2, 3, 224, 224))
o.shape
o = m.forward_features(torch.randn(2, 3, 224, 224))
print(o.shape)
