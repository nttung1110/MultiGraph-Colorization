import torch
import torch.nn.functional


def count_parameters(net):
    return sum(p.numel() for p in net.parameters() if p.requires_grad)


def l2_normalize(tensor, e=1e-8):
    norm = tensor.norm(p=2, dim=-1, keepdim=True).detach()
    out = tensor / (norm + e)
    return out


def get_coord_features(tensor):
    batch_size, num_channels, dim_y, dim_x = tensor.shape
    xx_ones = torch.ones([1, 1, 1, dim_x], dtype=torch.int32)
    yy_ones = torch.ones([1, 1, 1, dim_y], dtype=torch.int32)

    xx_range = torch.arange(dim_y, dtype=torch.int32)
    yy_range = torch.arange(dim_x, dtype=torch.int32)
    xx_range = xx_range[None, None, :, None]
    yy_range = yy_range[None, None, :, None]

    xx_channel = torch.matmul(xx_range, xx_ones)
    yy_channel = torch.matmul(yy_range, yy_ones)

    # transpose y
    yy_channel = yy_channel.permute(0, 1, 3, 2)

    xx_channel = xx_channel.float() / (dim_y - 1)
    yy_channel = yy_channel.float() / (dim_x - 1)

    xx_channel = xx_channel * 2 - 1
    yy_channel = yy_channel * 2 - 1

    xx_channel = xx_channel.repeat(batch_size, 1, 1, 1)
    yy_channel = yy_channel.repeat(batch_size, 1, 1, 1)

    out = torch.cat([xx_channel, yy_channel], dim=1).to(tensor.device)
    return out


def gather_by_label(out, labels):
    features = []
    unique_labels = torch.unique(labels, sorted=True).cpu().numpy().tolist()

    for label_index in unique_labels[:]:
        mask = (labels == label_index).unsqueeze(1).float()
        component_sum = torch.sum(mask).detach()

        component_features = out * mask
        component_features = torch.sum(component_features, dim=(2, 3))
        component_features = component_features / component_sum
        features.append(component_features)

    features = torch.stack(features, dim=1)
    return features


def gather_by_label_matrix(out, labels):
    max_label = torch.max(labels)
    labels = torch.unsqueeze(labels, dim=1)
    one_hot = torch.zeros([labels.shape[0], max_label + 1, labels.shape[2], labels.shape[3]]).to(out.device)
    one_hot = torch.scatter(one_hot, 1, labels, 1)[:, 1:, :, :]
    one_hot = one_hot.view(one_hot.shape[0], one_hot.shape[1], one_hot.shape[2] * one_hot.shape[3])

    out = out.view(out.shape[0], out.shape[1], out.shape[2] * out.shape[3]).permute(0, 2, 1)
    out = torch.bmm(one_hot, out)
    norm = torch.sum(one_hot, dim=-1).unsqueeze(-1)
    out = torch.div(out, norm)
    return out


def cosine_distance(output_a, output_b):
    distances = torch.nn.functional.cosine_similarity(output_a, output_b, dim=-1)
    distances = torch.ones_like(distances) - distances
    return distances


def squared_distance(output_a, output_b):
    distances = torch.sum((output_a - output_b).pow(2), dim=-1)
    return distances
