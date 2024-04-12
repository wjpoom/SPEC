import torch

from abc import ABCMeta, abstractmethod


class BaseWrapper(metaclass=ABCMeta):
    """
    This is the base model wrapper, if you want to evluate
    """

    @abstractmethod
    @torch.no_grad()
    def i2t_evaluate(self, subset_name, dataloader):
        pass

    @abstractmethod
    @torch.no_grad()
    def t2i_evaluate(self, subset_name, dataloader):
        pass

    @torch.no_grad()
    def evaluate(self, subset_name, dataloaders):
        """Computes the image-text matching scores and the image-to-text and text-to-image accuracy on a given subset
        Args:
            subset_name: the name of the subset
            dataloaders (Dict): include an "i2t_dataloader" and a "t2i_dataloader"
        Returns:
            scores(Dict of Tensor): `i2t_scores`, `t2i_scores`
            accuracy(Dict of Scalar): `i2t_accuracy`, t2i_accuracy`
        """
        # image to text retrieval
        i2t_scores, i2t_acc = self.i2t_evaluate(subset_name, dataloaders['i2t_dataloader'])
        print(f'{subset_name} subset: Image2Text Accuracy: {i2t_acc:.2f} %')

        # text to image retrieval
        t2i_scores, t2i_acc = self.t2i_evaluate(subset_name, dataloaders['t2i_dataloader'])
        print(f'{subset_name} subset: Text2Image Accuracy: {t2i_acc:.2f} %')

        """
        `i2t_scores`: tensor of shape NxL, N is the number of testing samples, L is the number of candidate texts per sample
        `t2i_scores`: tensor of shape NxK, N is the number of testing samples, K is the number of candidate images per sample
        """
        scores = {
            'i2t_scores': i2t_scores,
            't2i_scores': t2i_scores
        }

        accuracy = {
            'i2t_accuracy': i2t_acc,
            't2i_accuracy': t2i_acc,
        }

        return {
            "accuracy": accuracy,
            "scores": scores
        }
