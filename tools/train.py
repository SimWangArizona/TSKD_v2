import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import sys
sys.path.append("/data2/mdistiller-clean/mdistiller-master")
cudnn.benchmark = True

from mdistiller.models import cifar_model_dict, imagenet_model_dict
from mdistiller.distillers import distiller_dict
from mdistiller.dataset import get_dataset,get_dataset_strong,get_dataset_partial
from mdistiller.engine.utils import load_checkpoint, log_msg
from mdistiller.engine.cfg import CFG as cfg
from mdistiller.engine.cfg import show_cfg
from mdistiller.engine import trainer_dict


def main(cfg, resume, opts):
    experiment_name = cfg.EXPERIMENT.NAME
    if experiment_name == "":
        experiment_name = cfg.EXPERIMENT.TAG
    tags = cfg.EXPERIMENT.TAG.split(",")
    if opts:
        addtional_tags = ["{}:{}".format(k, v) for k, v in zip(opts[::2], opts[1::2])]
        tags += addtional_tags
        experiment_name += ",".join(addtional_tags)
    experiment_name = os.path.join(cfg.EXPERIMENT.PROJECT, experiment_name)
    if cfg.LOG.WANDB:
        try:
            import wandb

            wandb.init(project=cfg.EXPERIMENT.PROJECT, name=experiment_name, tags=tags)
        except:
            print(log_msg("Failed to use WANDB", "INFO"))
            cfg.LOG.WANDB = False

    # cfg & loggers
    show_cfg(cfg)
    # init dataloader & models
    if cfg.DISTILLER.TYPE == "TSKD_MKD":
        train_loader, val_loader, num_data, num_classes = get_dataset_strong(cfg)
    else:
        train_loader, val_loader, num_data, num_classes = get_dataset(cfg)

    # vanilla1 for patial data training
    if cfg.DISTILLER.TYPE == "NONE":
        if cfg.DATASET.TYPE == "imagenet":
            model_student = imagenet_model_dict[cfg.DISTILLER.STUDENT](pretrained=False)
        else:
            train_loader, val_loader, num_data, num_classes = get_dataset_partial(cfg)
            model_student = cifar_model_dict[cfg.DISTILLER.STUDENT][0](
                num_classes=num_classes
            )
        distiller = distiller_dict[cfg.DISTILLER.TYPE](model_student)
    # distillation
    else:
        print(log_msg("Loading teacher model", "INFO"))
        if cfg.DATASET.TYPE == "imagenet":
            model_teacher = imagenet_model_dict[cfg.DISTILLER.TEACHER](pretrained=True)
            model_student = imagenet_model_dict[cfg.DISTILLER.STUDENT](pretrained=False)
        else:
            net, pretrain_model_path = cifar_model_dict[cfg.DISTILLER.TEACHER]
            assert (
                pretrain_model_path is not None
            ), "no pretrain model for teacher {}".format(cfg.DISTILLER.TEACHER)
            model_teacher = net(num_classes=num_classes)
            model_teacher.load_state_dict(load_checkpoint(pretrain_model_path)["model"])
            model_student = cifar_model_dict[cfg.DISTILLER.STUDENT][0](
                num_classes=num_classes
            )
        model_student_normal = model_student
        model_teacher_normal = model_teacher

        if cfg.DISTILLER.TYPE == "TSKD_CAT":
            if cfg.DATASET.TYPE == "imagenet":
                model_dict_type = imagenet_model_dict
            else:
                model_dict_type = cifar_model_dict
            if cfg.DATASET.TYPE == "imagenet":
                model_teacher = model_dict_type[cfg.DISTILLER.TEACHER](pretrained=True)
                model_student = model_dict_type[cfg.DISTILLER.STUDENT](pretrained=False)
            else:
                if cfg.CAT_KD.teacher_dir is not None:
                    print('---------------custom teacher------------------')
                    net = model_dict_type[cfg.DISTILLER.TEACHER][0]
                    pretrain_model_path = cfg.CAT_KD.teacher_dir
                    print('get teacher dir from cfg')
                else:
                    print('---------------default teacher------------------')
                    net, pretrain_model_path = model_dict_type[cfg.DISTILLER.TEACHER]
                assert (
                        pretrain_model_path is not None
                ), "no pretrain model for teacher {}".format(cfg.DISTILLER.TEACHER)
                model_teacher = net(num_classes=num_classes)
                temp = load_checkpoint(pretrain_model_path)["model"]

                weights_dict = {}
                for k, v in temp.items():
                    new_k = k.replace('module.', '') if 'module' in k else k
                    weights_dict[new_k] = v
                temp = weights_dict

                temp = {k: v for k, v in temp.items() if k in model_teacher.state_dict()}

                model_teacher.load_state_dict(temp)
                model_student = model_dict_type[cfg.DISTILLER.STUDENT][0](
                    num_classes=num_classes
                )
            if cfg.if_test == True:

                if cfg.DISTILLER.TEACHER[-1] != 't':
                    print('converting teacher\'s structure')
                    if cfg.DATASET.TYPE == "imagenet":
                        tnet_test = model_dict_type[cfg.DISTILLER.TEACHER + '_test'](pretrained=False)
                    else:
                        tnet_test = model_dict_type[cfg.DISTILLER.TEACHER + '_test'][0](num_classes=num_classes)
                    model_dict = tnet_test.state_dict()

                    pretrained_dict = {k: v for k, v in model_teacher.state_dict().items() if k in model_dict}
                    if cfg.DATASET.TYPE == "imagenet":
                        pretrained_dict['conv_test.weight'] = model_teacher.state_dict()['fc.weight'].view(
                            model_teacher.fc.out_features, model_teacher.fc.in_features, 1, 1)
                    elif cfg.DISTILLER.TEACHER == 'ResNet50' or cfg.DISTILLER.TEACHER == 'WideResNet28x10_cifar100':
                        pretrained_dict['conv_test.weight'] = model_teacher.state_dict()['linear.weight'].view(
                            model_teacher.linear.out_features, model_teacher.linear.in_features, 1, 1)
                    elif cfg.DISTILLER.TEACHER == 'vgg13':
                        pretrained_dict['conv_test.weight'] = model_teacher.state_dict()['classifier.weight'].view(
                            model_teacher.classifier.out_features, model_teacher.classifier.in_features, 1, 1)
                    else:
                        pretrained_dict['conv_test.weight'] = model_teacher.state_dict()['fc.weight'].view(
                            model_teacher.fc.out_features, model_teacher.fc.in_features, 1, 1)
                    model_dict.update(pretrained_dict)
                    tnet_test.load_state_dict(model_dict)
                    model_teacher = tnet_test

                if cfg.DATASET.TYPE == "imagenet":
                    model_student = model_dict_type[cfg.DISTILLER.STUDENT + '_test'](pretrained=False)
                else:
                    model_student = model_dict_type[cfg.DISTILLER.STUDENT + '_test'][0](num_classes=num_classes)
        if cfg.DISTILLER.TYPE == "CRD":
            distiller = distiller_dict[cfg.DISTILLER.TYPE](
                model_student, model_teacher, cfg, num_data
            )

        else:
            # print(cfg)
            # print(distiller_dict[cfg.DISTILLER.TYPE])
            distiller = distiller_dict[cfg.DISTILLER.TYPE](
                model_student, model_teacher,cfg)

    distiller = torch.nn.DataParallel(distiller.cuda())

    if cfg.DISTILLER.TYPE != "NONE":
        print(
            log_msg(
                "Extra parameters of {}: {}\033[0m".format(
                    cfg.DISTILLER.TYPE, distiller.module.get_extra_parameters()
                ),
                "INFO",
            )
        )

    # train
    trainer = trainer_dict[cfg.SOLVER.TRAINER](
        experiment_name, distiller, train_loader, val_loader, cfg
    )
    trainer.train(resume=resume)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("training for knowledge distillation.")
    parser.add_argument("--cfg", type=str, default="../configs/cifar100/tskd/res32x4_res8x4.yaml")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)

    args = parser.parse_args()
    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    main(cfg, args.resume, args.opts)
