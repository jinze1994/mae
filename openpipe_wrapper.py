import tqdm
import torch
import openpipe
import numpy as np

import models_vit
from util.datasets import build_transform
from main_finetune import get_args_parser


@openpipe.register(pipelines="image-classification")
class MaeImagenet1K(torch.nn.Module):

    def __init__(self, args):
        super().__init__()
        self.transform = build_transform(0, args)
        self.model = models_vit.__dict__[args.model](
            num_classes=args.nb_classes,
            drop_path_rate=args.drop_path,
            global_pool=args.global_pool,
        )
        self.id2label, _ = openpipe.datasets.ImageNet1K.vit_label_mapping()
        if args.resume:
            if args.resume.startswith('http'):
                checkpoint = torch.hub.load_state_dict_from_url(
                    args.resume, map_location='cpu', check_hash=True)
            else:
                checkpoint = torch.load(args.resume, map_location='cpu')
            self.model.load_state_dict(checkpoint['model'])
            print("Resume checkpoint %s" % args.resume)

    def preprocess(self, *args, **kwargs):
        output = self.transform(kwargs['images'])
        output = output.unsqueeze(0)
        return {"pixel_values": output}

    def __call__(self, *args, **kwargs):
        logits = self.model(kwargs['pixel_values'])
        return {"logits": logits}

    def postprocess(self, *args, **kwargs):
        logits = kwargs["logits"]
        top_k = kwargs.pop('top_k', 5)
        probs = logits.softmax(-1)[0]
        scores, ids = probs.topk(top_k)

        scores = scores.tolist()
        ids = ids.tolist()
        return [{"score": score, "label": self.id2label[_id]} for score, _id in zip(scores, ids)]


if __name__ == '__main__':
    parser = get_args_parser()
    given_args = '--eval --resume https://dl.fbaipublicfiles.com/mae/finetune/mae_finetuned_vit_base.pth --model vit_base_patch16'
    args = parser.parse_args(given_args.split())

    model = MaeImagenet1K(args)
    dataset = openpipe.datasets.ImageNet1K(split='val')
    iterator = openpipe.launch(
        'image-classification',
        dataset=dataset,
        model=model)

    gt, pred = [], []
    for i, res in enumerate(tqdm.tqdm(iterator, total=dataset.global_data_size)):
        pred.append(res['image'][0]['label'])
        gt.append(res['label'])
        if i > 100:
            break

    acc1 = (np.array(gt) == np.array(pred)).mean()
    print(acc1)
