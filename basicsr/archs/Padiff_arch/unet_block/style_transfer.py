import torch
import torchvision

def gram_matrix(input):
    b, c, h, w = input.size()
    features = input.view(b * c, h * w)
    G = torch.mm(features, features.t())
    return G.div(b * c * h * w)

class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, resize=False):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[4:9].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[9:16].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[16:23].eval())
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False

        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.mean = torch.nn.Parameter(torch.tensor([0.485, 0.456, 0.406], device='cuda').view(1, 3, 1, 1))
        self.std = torch.nn.Parameter(torch.tensor([0.229, 0.224, 0.225], device='cuda').view(1, 3, 1, 1))
        self.resize = resize

    def forward(self, input, support, support2, style):
        # print(input.shape, '--', target.shape, '--', style.shape)
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            support = support.repeat(1, 3, 1, 1)
            support2 = support2.repeat(1, 3, 1, 1)
            # target = target.repeat(1, 3, 1, 1)
            style = style.repeat(1, 3, 1, 1)

        # input = (input + 1) / 2
        # target = (target + 1) / 2
        # style = (style + 1) / 2
        # input = (input - self.mean) / self.std
        # target = (target - self.mean) / self.std
        # style = (style - self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            # target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
            style = self.transform(style, mode='bilinear', size=(224, 224), align_corners=False)
        content_loss = 0.0
        style_loss = 0.0
        style_loss2 = 0.0
        x = input
        y = support
        y2 = support2
        # y = target
        s = style
        for block in self.blocks:
            x = block(x)
            y = block(y)
            y2 = block(y2)
            s = block(s)
            # content_loss += torch.nn.functional.mse_loss(x, y)
            style_loss += torch.nn.functional.mse_loss(gram_matrix(x), gram_matrix(s).detach())
            style_loss2 += torch.nn.functional.mse_loss(gram_matrix(y), gram_matrix(s).detach()) \
                           + torch.nn.functional.mse_loss(gram_matrix(y2), gram_matrix(s).detach())

        # return content_loss, style_loss
        return style_loss + style_loss2