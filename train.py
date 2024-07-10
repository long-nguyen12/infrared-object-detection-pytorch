from utils.data import *
from utils.metric import *
from utils.lr_scheduler import *
from argparse import ArgumentParser
import torch
from model.net import *
from model.loss import *
from tqdm import tqdm
import os.path as osp
import os
import time
from glob import glob
import albumentations as A


def parse_args():
    parser = ArgumentParser(description="Implement of model")

    parser.add_argument("--train_path", type=str, default="data/NUST-SIRST")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--lr", type=float, default=1e-4)

    parser.add_argument("--base-size", type=int, default=256)
    parser.add_argument("--crop-size", type=int, default=256)

    parser.add_argument("--mode", type=str, default="train")
    args = parser.parse_args()
    return args


class Trainer(object):
    def __init__(self, args):
        assert args.mode == "train" or args.mode == "test"

        self.args = args
        self.start_epoch = 0
        self.mode = args.mode

        dataset = args.train_path.split("/")[-1]
        self.save_folder = f"snapshots/{dataset}"

        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder, exist_ok=True)
        else:
            print("Save path existed")

        train_img_paths = []
        train_mask_paths = []
        train_img_paths = glob("{}/training/images/*".format(args.train_path))
        train_mask_paths = glob("{}/training/masks/*".format(args.train_path))
        train_img_paths.sort()
        train_mask_paths.sort()

        test_img_paths = []
        test_mask_paths = []
        test_img_paths = glob("{}/test/images/*".format(args.train_path))
        test_mask_paths = glob("{}/test/masks/*".format(args.train_path))
        test_img_paths.sort()
        test_mask_paths.sort()

        transform = A.Compose(
            [
                A.Resize(height=256, width=256),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )
        mask_transform = A.Compose(
            [
                A.Resize(height=256, width=256),
            ]
        )

        train_dataset = Dataset(
            train_img_paths,
            train_mask_paths,
            transform=transform,
            mask_transform=mask_transform,
        )

        self.train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
        )

        test_dataset = Dataset(
            test_img_paths,
            test_mask_paths,
            transform=transform,
            mask_transform=mask_transform,
        )

        self.val_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            pin_memory=True,
            drop_last=True,
        )

        device = torch.device("cuda")
        self.device = device

        # Model
        model = SegmentNet()
        model.to(device)
        self.model = model

        # Optimizer and Scheduler
        params = model.parameters()
        self.optimizer = torch.optim.Adam(params, args.lr)
        self.scheduler = LR_Scheduler_Head(
            "poly", args.lr, args.epochs, len(self.train_loader), lr_step=10
        )

        # Loss funcitons
        self.loss_fun = SoftLoULoss()

        # Metrics
        self.PD_FA = PD_FA(1, 10, args.base_size)
        self.mIoU = mIoU(1)
        self.ROC = ROCMetric(1, 10)
        self.best_iou = 0

    def train(self, epoch):
        self.model.train()
        tbar = tqdm(self.train_loader)
        losses = AverageMeter()

        for i, (data, mask) in enumerate(tbar):

            data = data.to(self.device)
            labels = mask.to(self.device)

            masks, pred = self.model(data)
            loss = 0

            loss = loss + self.loss_fun(pred, labels)
            for j in range(len(masks)):
                loss = loss + self.loss_fun(masks[j], labels)

            loss = loss / (len(masks) + 1)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            losses.update(loss.item(), pred.size(0))
            tbar.set_description("Epoch %d, loss %.4f" % (epoch, losses.avg))

    def test(self, epoch):
        self.model.eval()
        self.mIoU.reset()
        self.PD_FA.reset()
        tbar = tqdm(self.val_loader)
        with torch.no_grad():
            for i, (data, mask) in enumerate(tbar):

                data = data.to(self.device)
                mask = mask.to(self.device)

                _, pred = self.model(data)

                self.mIoU.update(pred, mask)
                self.PD_FA.update(pred, mask)
                self.ROC.update(pred, mask)
                _, mean_IoU = self.mIoU.get()

                tbar.set_description("Epoch %d, IoU %.4f" % (epoch, mean_IoU))
            FA, PD = self.PD_FA.get(len(self.val_loader))
            _, mean_IoU = self.mIoU.get()
            ture_positive_rate, false_positive_rate, _, _ = self.ROC.get()

            if self.mode == "train":
                if mean_IoU > self.best_iou:
                    self.best_iou = mean_IoU

                    torch.save(self.model.state_dict(), self.save_folder + "/best.pth")
                    with open(osp.join(self.save_folder, "metric.log"), "a") as f:
                        f.write(
                            "{} - {:04d}\t - IoU {:.4f}\t - PD {:.4f}\t - FA {:.4f}\n".format(
                                time.strftime(
                                    "%Y-%m-%d-%H-%M-%S", time.localtime(time.time())
                                ),
                                epoch,
                                self.best_iou,
                                PD[0],
                                FA[0] * 1000000,
                            )
                        )

                all_states = {
                    "net": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "epoch": epoch,
                    "iou": self.best_iou,
                }
                torch.save(all_states, self.save_folder + "/checkpoint.pth")
            elif self.mode == "test":
                print("mIoU: " + str(mean_IoU) + "\n")
                print("Pd: " + str(PD[0]) + "\n")
                print("Fa: " + str(FA[0] * 1000000) + "\n")


if __name__ == "__main__":
    args = parse_args()

    trainer = Trainer(args)
    if trainer.mode == "train":
        for epoch in range(trainer.start_epoch, args.epochs):
            trainer.train(epoch)
            trainer.test(epoch)
    else:
        trainer.test(1)
