import tqdm
import torch
import openpipe
import numpy as np

import models_vit
from util.datasets import build_transform
from main_finetune import get_args_parser


class MaeClassificationTask(openpipe.tasks.ImageClassificationTask):

    def __init__(self, args):
        super().__init__()
        self.transform = build_transform(0, args)
        self.model = models_vit.__dict__[args.model](
            num_classes=args.nb_classes,
            drop_path_rate=args.drop_path,
            global_pool=args.global_pool,
        )
        if args.resume:
            if args.resume.startswith('http'):
                checkpoint = torch.hub.load_state_dict_from_url(
                    args.resume, map_location='cpu', check_hash=True)
            else:
                checkpoint = torch.load(args.resume, map_location='cpu')
            self.model.load_state_dict(checkpoint['model'])
            print("Resume checkpoint %s" % args.resume)

    def preprocess(self, image):
        return self.transform(image).unsqueeze(0)

    def forward(self, inputs):
        return {"logits": self.model(inputs)}

    def id2label(self):
        with open('id2label.txt') as f:
            return f.read().split('\n')[1 : 1001]


if __name__ == '__main__':
    parser = get_args_parser()
    given_args = '--eval --resume https://dl.fbaipublicfiles.com/mae/finetune/mae_finetuned_vit_base.pth --model vit_base_patch16'
    args = parser.parse_args(given_args.split())
    task = MaeClassificationTask(args)

    dataset = openpipe.datasets.ImageNet1K(split='val')
    metric = openpipe.metrics.Accuracy()
    pip = openpipe.pipeline.concat(dataset, task, metric)
    print(pip)

    acc1 = []
    for i, res in enumerate(tqdm.tqdm(pip, total=dataset.global_data_size)):
        acc1.append(res)
        if i == 100:
            break

    acc1 = np.mean(acc1)
    print(acc1)
