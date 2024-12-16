import argparse
import torch
import torch.backends.cudnn as cudnn
import sys
sys.path.append("/data2/mdistiller-clean/mdistiller-master")
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
cudnn.benchmark = True
import numpy as np
from mdistiller.distillers import Vanilla
from mdistiller.models import cifar_model_dict, imagenet_model_dict
from mdistiller.dataset import get_dataset
from mdistiller.dataset.imagenet import get_imagenet_val_loader
from mdistiller.engine.utils import load_checkpoint, validate
from mdistiller.engine.cfg import CFG as cfg



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, default="resnet20")
    parser.add_argument("-c", "--ckpt", type=str, default="pretrain")
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        default="cifar100",
        choices=["cifar100", "imagenet"],
    )
    parser.add_argument("-bs", "--batch-size", type=int, default=64)
    args = parser.parse_args()

    cfg.DATASET.TYPE = args.dataset
    cfg.DATASET.TEST.BATCH_SIZE = args.batch_size
    train_loader, val_loader, num_data, num_classes = get_dataset(cfg)
    base_model, _ = cifar_model_dict[args.model]
    commonprefix = "/data2/mdistiller-clean/mdistiller-master/output/cifar100_baselines/"

    prefix1 = "vanilla1,res20/student_"
    prefix2 = "vanilla2,res20/student_"
    prefix3 = "vanilla3,res20/student_"

    model_time_stamp = np.arange(5,65,5)
    model = base_model(num_classes=num_classes)
    model2 = base_model(num_classes=num_classes)

# load vanilla1 and vanilla2
    diff_list = []
    for idx in model_time_stamp:
        pretrain_model_path = commonprefix + prefix1 + str(idx)
        pretrain_model_path2 = commonprefix + prefix2 + str(idx)
        ckpt = pretrain_model_path
        ckpt2 = pretrain_model_path2
        model.load_state_dict(load_checkpoint(ckpt)["model"])
        model2.load_state_dict(load_checkpoint(ckpt2)["model"])

        param1 = list(model.parameters())
        param2 = list(model2.parameters())
        diff_sum = 0
        for (p1,p2) in zip(param1,param2):
            diff = p1-p2
            diff_sum += torch.norm(diff, p=2)
        diff_list.append(diff_sum)
    diff_list = [tensor.detach().numpy() for tensor in diff_list]
    # print(diff_list)
    import matplotlib.pyplot as plt

    plt.plot(model_time_stamp, diff_list,label = 'parameter difference during training')
    plt.title('Norm-2 vs. Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Norm-2')
    plt.legend()
    plt.show()
    train_loss_values1 = []
    # 打开文件
    with open('/data2/mdistiller-clean/mdistiller-master/output/cifar100_baselines/vanilla1,res20/worklog.txt', 'r') as file:
        # 遍历文件的每一行
        for line in file:
            # 如果该行以'train_loss:'开头
            if line.startswith('train_loss:'):
                # 输出train_loss后面的内容（去除首尾空格）
                value = line.split('train_loss:')[1].strip()
                train_loss_values1.append(float(value))
    print(train_loss_values1)
    plt.plot(np.arange(1,61),train_loss_values1,label='resnet20-1')


    train_loss_values2 = []
    # 打开文件
    with open('/data2/mdistiller-clean/mdistiller-master/output/cifar100_baselines/vanilla2,res20/worklog.txt',
              'r') as file:
        # 遍历文件的每一行
        for line in file:
            # 如果该行以'train_loss:'开头
            if line.startswith('train_loss:'):
                # 输出train_loss后面的内容（去除首尾空格）
                value = line.split('train_loss:')[1].strip()
                train_loss_values2.append(float(value))
    print(train_loss_values2)
    plt.plot(np.arange(1, 61), train_loss_values2,label='resnet20-2')
    plt.title('Loss vs. Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # # print(list(model.parameters()))
    # model = Vanilla(model)
    # model = model.cuda()
    # model = torch.nn.DataParallel(model)
    # test_acc, test_acc_top5, test_loss = validate(val_loader, model)
    #
    # model2 = Vanilla(model2)
    # model2 = model2.cuda()
    # model2 = torch.nn.DataParallel(model2)
    # test_acc2, test_acc_top52, test_loss2 = validate(val_loader, model2)


