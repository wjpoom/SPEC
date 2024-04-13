import argparse
import os
import torch
import random
import numpy as np

from models import get_model
from dataset import get_data


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def main():
    os.makedirs(args.out_path, exist_ok=True)

    # set random seeds
    seed_all(args.seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # load model
    model, image_preprocess = get_model(model_name=args.model_name,
                                        cache_dir=args.model_cache_dir,
                                        device=device)

    # load data
    data = get_data(data_root=args.data_root,
                    subset_names=args.subset_names,
                    image_preprocess=image_preprocess,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers)

    # evaluate on each subset
    print(f'\nBegin the evaluation of {args.model_name} on all selected subsets.')
    result = {}
    i2t_acc = 0.
    t2i_acc = 0.
    subset_num = 0
    for subset_name, dataloaders in data.items():
        subset_result = model.evaluate(subset_name=subset_name, dataloaders=dataloaders)
        result[subset_name] = subset_result
        i2t_acc += subset_result['accuracy']['i2t_accuracy']
        t2i_acc += subset_result['accuracy']['t2i_accuracy']
        subset_num += 1
    print(f'\nFinished the evaluation of {args.model_name} on all selected subsets.')
    print(f'average all subset: Image2Text Accuracy: {i2t_acc/subset_num:.2f} %')
    print(f'average all subset: Text2Image Accuracy: {t2i_acc/subset_num:.2f} %')

    # save results
    out_fn = f"{args.model_name}__evaluate_result.pth"
    out_fn = os.path.join(args.out_path, out_fn)
    torch.save(result, out_fn)
    print(f'result saved to {out_fn}.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Vision Language Models Evaluation Pipeline')
    parser.add_argument('--model-name',
                        type=str,
                        default='clip')
    parser.add_argument('--pretrained',
                        type=str,
                        help="the pretrained model checkpoint")
    parser.add_argument('--model-cache-dir',
                        type=str,
                        default='~/.cache',
                        help='the path to cache the downloaded model checkpoints')
    parser.add_argument('--subset-names',
                        type=str,
                        nargs='+',
                        choices=['count', 'relative_size', 'absolute_size', 'relative_spatial', 'absolute_spatial',
                                 'existence'],
                        help='type of generated dataset type for enhanced ability')
    parser.add_argument('--data-root',
                        type=str,
                        help='the path the the root dir of data')
    parser.add_argument('--out-path',
                        type=str,
                        default='out',
                        help="path to save evaluation-real results")
    parser.add_argument('--batch-size',
                        type=int,
                        default=64)
    parser.add_argument('--num-workers',
                        type=int,
                        default=8)
    parser.add_argument('--seed',
                        type=int,
                        default=1,
                        help="random seed for reproducibility")

    args = parser.parse_args()

    main()
