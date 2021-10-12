import torch
from vit import ViT
import torch
import torchvision
import torchvision.transforms as transform
from torch import optim
from metric import Accuracy
import time
from config import configs


def seed():
    torch.manual_seed(0)
    torch.set_default_tensor_type("torch.FloatTensor")


def main():
    seed()
    for i in range(3):
        for config in configs:
            config["lr"] = config["batch_size"] * 1e-4
            config["log_interval"] = 1
            train(config=config)


def train(config=None):
    ttic = time.time()
    print(config)
    if config == None:
        assert "no config"
    epoch_num = 16
    # epoch_num = 1

    batch_size = config["batch_size"]
    momentum = config["momentum"]
    dim = config["dim"]
    depth = config["depth"]
    heads = config["heads"]
    mlp_dim = config["mlp_dim"]
    lr = config["lr"]
    log_interval = config["log_interval"]

    model = ViT(
        image_size=28,  # fixed
        patch_size=4,  # fixed
        num_classes=10,  # fixed
        emb_dropout=0.1,  # fixed
        channels=1,  # fixed
        dim=dim,  # after embedding dim
        depth=depth,  # depth of multi head
        heads=heads,  # head num
        mlp_dim=mlp_dim,  # feed forward hidden layer dim
    )
    # print(model)

    train_transforms = transform.Compose([transform.ToTensor()])
    train_dataset = torchvision.datasets.MNIST(
        "./data", train=True, transform=train_transforms
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False, drop_last=True
    )

    test_transforms = transform.Compose([transform.ToTensor()])
    test_dataset = torchvision.datasets.MNIST(
        "./data", train=False, transform=test_transforms
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=256, shuffle=False, drop_last=True
    )

    optimizer = optim.SGD(params=model.parameters(), lr=lr, momentum=momentum)
    loss_fn = torch.nn.CrossEntropyLoss()

    train_metric = Accuracy()
    acc_top1 = Accuracy()

    for epoch_idx in range(epoch_num):
        # train
        train_metric.reset()
        model.train()
        btic = time.time()
        # tic = time.time()
        for step_idx, (data, target) in enumerate(train_dataloader):
            batch_st = time.perf_counter()
            optimizer.zero_grad()

            # st = time.perf_counter()
            output = model(data)
            # ed = time.perf_counter()
            # print(f'model forward timing: {ed -st}')

            loss = loss_fn(output, target)

            # st = time.perf_counter()
            loss.backward()
            # ed = time.perf_counter()
            # print(f'model backward timing: {ed -st}')

            # st = time.perf_counter()
            optimizer.step()
            # ed = time.perf_counter()
            # print(f'optimizer timing: {ed -st}')

            batch_ed = time.perf_counter()
            print(f'one batch timing: {batch_ed - batch_st}')
           
            train_metric.update(target.cpu(), output.cpu())
            if log_interval and not (step_idx + 1) % log_interval:
                train_metric_name, train_metric_score = train_metric.get()
                step_throughput = batch_size * log_interval / (time.time() - btic)
                btic = time.time()
                print(
                    f"Epoch[{epoch_idx}] Batch [{step_idx+1}]\t"
                    f"Speed: {int(step_throughput)} samples/sec\t"
                    f"{train_metric_name}={train_metric_score:.8f}\t"
                    f"loss={loss.cpu().item():.8f}"
                )

        # eval
        acc_top1.reset()
        model.eval()
        with torch.no_grad():
            for i, (images, target) in enumerate(test_dataloader):
                output = model(images)
                acc_top1.update(target.cpu(), output.cpu())
        acc_top1_name, acc_top1_score = acc_top1.get()
        print(f"[Epoch {epoch_idx}] valid:{acc_top1_name}={acc_top1_score}")

    all_time = time.time() - ttic


if __name__ == "__main__":
    main()
