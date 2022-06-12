import torch


class MaskClassifier(torch.nn.Module):
    def __init__(self, backbone_name="efficientnet_b4"):
        super(MaskClassifier, self).__init__()
        
        self.backbone_name = backbone_name

        if self.backbone_name == "efficientnet_b4":
            self.backbone = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_b4', pretrained=True)
            utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_convnets_processing_utils')
            # efficientnet.eval().to(device)

            for p in self.backbone.parameters():
                p.requires_grad = False

            self.backbone.classifier = torch.nn.Sequential(
                torch.nn.AdaptiveAvgPool2d(output_size=1),
                torch.nn.Flatten(),
                torch.nn.Dropout(p=0.4, inplace=False),
                torch.nn.Linear(in_features=1792, out_features=2, bias=True)
            )
        else:
            raise ValueError("Please provide an allowed backbone name")

    def forward(self, x):
        x = self.backbone(x)

        return x


if __name__ == "__main__":
    model = MaskClassifier()
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(params)
