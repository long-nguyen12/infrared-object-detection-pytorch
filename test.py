import os
from argparse import ArgumentParser

import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

from model.loss import *
from model.net import *
from utils.data import *
from utils.lr_scheduler import *
from utils.metric import *

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def parse_args():
    parser = ArgumentParser(description="Implement of model")

    parser.add_argument("--train_path", type=str, default="data/IRSTD-1k")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--lr", type=float, default=0.05)

    parser.add_argument("--base-size", type=int, default=256)
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--warm-epoch", type=int, default=5)

    parser.add_argument("--mode", type=str, default="test")
    args = parser.parse_args()
    return args


def total_visulization_generation(
    dataset_dir, test_txt, suffix, target_image_path, target_dir
):
    source_image_path = dataset_dir + "/images"

    txt_path = test_txt
    ids = []
    with open(txt_path, "r") as f:
        ids += [line.strip() for line in f.readlines()]

    for i in range(len(ids)):
        source_image = source_image_path + "/" + ids[i] + suffix
        target_image = target_image_path + "/" + ids[i] + suffix
        shutil.copy(source_image, target_image)
    for i in range(len(ids)):
        source_image = target_image_path + "/" + ids[i] + suffix
        img = Image.open(source_image)
        img = img.resize((256, 256), Image.Resampling.LANCZOS)
        img.save(source_image)
    for m in range(len(ids)):
        plt.figure(figsize=(10, 6))
        plt.subplot(1, 3, 1)
        img = plt.imread(target_image_path + "/" + ids[m] + suffix)
        plt.imshow(img, cmap="gray")
        plt.xlabel("Raw Image", size=11)

        plt.subplot(1, 3, 2)
        img = plt.imread(target_image_path + "/" + ids[m] + "_GT" + suffix)
        plt.imshow(img, cmap="gray")
        plt.xlabel("Ground Truth", size=11)

        plt.subplot(1, 3, 3)
        img = plt.imread(target_image_path + "/" + ids[m] + "_Pred" + suffix)
        plt.imshow(img, cmap="gray")
        plt.xlabel("Predicts", size=11)

        plt.savefig(
            target_dir + "/" + ids[m].split(".")[0] + "_fuse" + suffix,
            facecolor="w",
            edgecolor="red",
        )


def save_Pred_GT(pred, labels, target_image_path, img_name):

    predsss = np.array((pred > 0).cpu()).astype("int64") * 255
    predsss = np.uint8(predsss)
    labelsss = labels * 255
    labelsss = np.uint8(labelsss.cpu())

    img = Image.fromarray(predsss.reshape(256, 256))
    img.save(target_image_path + "/" + img_name)
    img = Image.fromarray(labelsss.reshape(256, 256))
    img.save(target_image_path + "/" + img_name.replace(".png", "_GT.png"))


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

        valset = IRSTD_Dataset_Test(args, mode="val", dataset=dataset)

        self.val_loader = Data.DataLoader(valset, 1, drop_last=False)
        device = torch.device("cpu")
        self.device = device

        # Model
        model = SegmentNet()
        model.to(device)
        self.model = model

        # Metrics
        self.PD_FA = PD_FA(1, 255, args.base_size)
        self.mIoU = mIoU(1)
        self.ROC = ROCMetric(1, 10)
        self.best_iou = 0
        self.warm_epoch = args.warm_epoch

        weight = torch.load(f"{self.save_folder}/best.pth", map_location="cpu")
        self.model.load_state_dict(weight, strict=True)
        self.warm_epoch = -1

        self.visulization_path = "result_image/test"
        self.visulization_fuse_path = "result_image/test"

        if not os.path.exists(self.visulization_path):
            os.makedirs(self.visulization_path, exist_ok=True)

    def test(self, epoch):
        self.model.eval()
        self.mIoU.reset()
        self.PD_FA.reset()
        tbar = tqdm(self.val_loader)
        tag = False
        best_IoU = 0.0
        num = 0

        with torch.no_grad():
            for i, (data, mask, img_name) in enumerate(tbar):

                data = data.to(self.device)
                mask = mask.to(self.device)

                if epoch > self.warm_epoch:
                    tag = False

                _, pred = self.model(data, tag)

                self.mIoU.update(pred, mask)
                self.PD_FA.update(pred, mask)
                self.ROC.update(pred, mask)
                _, mean_IoU = self.mIoU.get()

                if mean_IoU > best_IoU:
                    best_IoU = mean_IoU
                    print(best_IoU, img_name)

                tbar.set_description("Epoch %d, IoU %.4f" % (epoch, mean_IoU))
                img_name = str(round(mean_IoU, 5)) + "_" + img_name[0] + ".png"
                save_Pred_GT(pred, mask, self.visulization_path, img_name)
                num += 1
            FA, PD = self.PD_FA.get(len(self.val_loader))
            _, mean_IoU = self.mIoU.get()
            # total_visulization_generation(
            #     args.train_path,
            #     f"{args.train_path}/test.txt",
            #     ".png",
            #     self.visulization_path,
            #     self.visulization_fuse_path,
            # )
            tpr, fpr, _, _ = self.ROC.get()
            
            plt.plot(fpr,tpr,label="AUC")
            plt.ylabel('True Positive Rate')
            plt.xlabel('False Positive Rate')
            plt.legend(loc=4)
            plt.show()
            
            print("mIoU: " + str(mean_IoU) + "\n")
            print("Pd: " + str(PD[0]) + "\n")
            print("Fa: " + str(FA[0] * 1000000) + "\n")


if __name__ == "__main__":
    args = parse_args()

    trainer = Trainer(args)
    trainer.test(1)
