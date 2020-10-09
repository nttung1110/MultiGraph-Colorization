import os
import glob
import numpy as np
import torch
import torch.utils.data
import torch.optim as optim
import torch.nn.functional
from natsort import natsorted
from loader.painting_loader import PairAnimeDataset, loader_collate
from models.new_shallow_net import UNet
from models.utils import cosine_distance


def compute_loss(output_a, output_b, positive_pairs, colors_a, colors_b, k=0.6):
    pos_distances, neg_distances = [], []
    colors_b = colors_b[0, ...]

    for index in range(0, positive_pairs.shape[1]):
        index_a = positive_pairs[0, index, 0]
        index_b = positive_pairs[0, index, 1]

        item_a = output_a[:, index_a, :]
        item_a = item_a.unsqueeze(1).repeat([1, output_b.shape[1], 1])
        current_color = colors_a[0, index_a, :]

        distances = cosine_distance(item_a, output_b)
        assert not torch.isnan(distances).any()
        pos_distances.append(distances[0, index_b])

        same_color_indices = torch.sum(colors_b == current_color, dim=-1)
        same_color_indices = torch.nonzero(same_color_indices).squeeze(-1)

        num_anchors = min(4, distances.shape[1] - same_color_indices.shape[0])
        if num_anchors == 0:
            continue
        ignore_pos = torch.zeros_like(distances)
        ignore_pos[0, same_color_indices] = 2.0

        min_distances = torch.topk(distances + ignore_pos, k=num_anchors, largest=False, dim=-1)[0]
        neg_distances.append(min_distances)

    pos_distance = torch.mean(torch.stack(pos_distances, dim=-1))
    loss = pos_distance

    if len(neg_distances) > 0:
        neg_distances = torch.cat(neg_distances, dim=-1)
        neg_loss = torch.clamp(torch.full_like(neg_distances, k) - neg_distances, min=0.0)
        neg_loss = torch.mean(neg_loss)
        loss = loss + neg_loss
    return loss


def load_existing_checkpoint(model, optimizer, config):
    weight_paths = natsorted(glob.glob(os.path.join(config.spot_checkpoint, "*.pth")))

    if len(weight_paths) == 0:
        return 0

    weight_path = weight_paths[-1]
    weight_data = torch.load(weight_path)

    model.load_state_dict(weight_data["model"])
    optimizer.load_state_dict(weight_data["optimizer"])

    epoch = int(os.path.basename(weight_path)[6:9]) + 1
    return epoch


def save_checkpoint(model, optimizer, config, epoch):
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    path = os.path.join(config.spot_checkpoint, "model_%03d.pth" % epoch)
    torch.save(state, path)


def main(config):
    device = config.device

    image_size = (768, 512)
    mean = [2.0, 7.0, 20.0, 20.0, 10.0, 0.0, 0.0, 0.0]
    std = [0.8, 2.0, 10.0, 10.0, 30.0, 20.0, 30.0, 1.0]
    mean = np.array(mean)[:, np.newaxis][:, np.newaxis]
    std = np.array(std)[:, np.newaxis][:, np.newaxis]

    # Dataset
    dataset = PairAnimeDataset(config.data_folder, image_size, mean, std)
    print("Dataset size:", len(dataset))
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=config.batch_size, shuffle=True, collate_fn=loader_collate)

    # Model
    model = UNet(8, 0.0, mode="train")
    model.to(device)

    # Optimization
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, config.step_size, gamma=0.1)

    # Load the checkpoint
    start_epoch = load_existing_checkpoint(model, optimizer, config)

    # Training loop
    for epoch in range(start_epoch, config.num_epochs):
        print("===============")
        print("Epoch:", epoch)
        model.train()
        bug_count = 0

        for step, data in enumerate(loader):
            list_a, list_b, positive_pairs = data
            features_a, mask_a, colors_a, components_a = list_a
            features_b, mask_b, colors_b, components_b = list_b

            features_a = features_a.float().to(device)
            mask_a = mask_a.long().to(device)
            features_b = features_b.float().to(device)
            mask_b = mask_b.long().to(device)
            positive_pairs = positive_pairs.long().to(device)
            colors_a = colors_a.to(device)
            colors_b = colors_b.to(device)

            # check for bug
            if positive_pairs.shape[1] == 0:
                bug_count += 1
                continue

            # run the model
            output_a, x_a, y_a = model(features_a, mask_a)
            output_b, x_b, y_b = model(features_b, mask_b)

            loss_out = compute_loss(output_a, output_b, positive_pairs, colors_a, colors_b)
            loss_x = compute_loss(x_a, x_b, positive_pairs, colors_a, colors_b)
            loss_y = compute_loss(y_a, y_b, positive_pairs, colors_a, colors_b)
            loss = loss_out + 0.4 * loss_x + 0.4 * loss_y

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % config.print_freq == 0:
                print("Loss: %.4f    X: %.4f    Y: %.4f" % (loss_out.item(), loss_x.item(), loss_y.item()))

        if epoch % config.save_freq == 0:
            save_checkpoint(model, optimizer, config, epoch)
        scheduler.step(epoch)
        print("Number of bug files: %d" % bug_count)
    print("Finished")


if __name__ == "__main__":
    from loader.config_utils import load_config, convert_to_object

    main(convert_to_object(load_config("D:/Documents/Cinnamon/painting/hades_painting/local_config.ini")))
