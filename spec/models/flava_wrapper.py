import torch
import numpy as np
from tqdm import tqdm
from transformers import FlavaProcessor, FlavaForPreTraining, BertTokenizer, FlavaFeatureExtractor
from .base_wrapper import BaseWrapper


class FlavaWrapper(BaseWrapper):
    def __init__(self, cache_dir, device):
        # load model, tokenizer, processor
        self.model = FlavaForPreTraining.from_pretrained("facebook/flava-full", cache_dir=cache_dir).eval()
        self.model = self.model.to(device)
        self.feature_extractor = FlavaFeatureExtractor.from_pretrained("facebook/flava-full", cache_dir=cache_dir)
        self.tokenizer = BertTokenizer.from_pretrained("facebook/flava-full", cache_dir=cache_dir)
        self.processor = FlavaProcessor.from_pretrained("facebook/flava-full", cache_dir=cache_dir)
        self.device = device

    @torch.no_grad()
    def i2t_evaluate(self, subset_name, dataloader):
        tqdm_i2t_loader = tqdm(dataloader)
        tqdm_i2t_loader.set_description(f"Image to Text retrieval on <{subset_name}>")
        i2t_scores = []
        i2t_correct_num = 0
        total_num = 0
        for batch in tqdm_i2t_loader:
            bs = len(batch['label'])
            # get query images
            query_images = batch['query_image']  # [B x PIL.Image] (B:batch size)

            # compute normalized image embeddings
            inputs = self.feature_extractor(images=query_images, return_tensors="pt").to(self.device)
            image_embeddings = self.model.flava.get_image_features(**inputs).cpu().numpy()[:, 0, :]
            image_embeddings = image_embeddings / np.linalg.norm(image_embeddings, axis=1, keepdims=True)
            image_embeddings = torch.tensor(image_embeddings)   # (B, D)

            # get candidate texts
            candidate_texts = batch['candidate_texts']   # B, L

            # compute normalized text embeddings
            text_embeddings = []
            for texts in candidate_texts:
                text_input = self.tokenizer(text=texts, return_tensors="pt", padding="max_length", max_length=77).to(self.device)
                text_feats = self.model.flava.get_text_features(**text_input).cpu().numpy()[:, 0, :]
                text_feats = text_feats / np.linalg.norm(text_feats, axis=1, keepdims=True)
                text_feats = torch.tensor(text_feats)
                text_embeddings.append(text_feats)
            text_embeddings = torch.stack(text_embeddings, dim=0)   # (B, L, D)

            # calculate matching result
            batch_i2t_scores = torch.einsum('BD,BLD->BL', [image_embeddings, text_embeddings]).cpu()
            i2t_scores.append(batch_i2t_scores)
            gt_labels = batch['label']
            pred_labels = batch_i2t_scores.argmax(dim=-1)
            correct_num = (gt_labels == pred_labels).sum()
            i2t_correct_num += correct_num.item()
            total_num += bs

        i2t_scores = torch.cat(i2t_scores, dim=0)
        i2t_acc = 100 * i2t_correct_num / total_num

        return i2t_scores, i2t_acc

    @torch.no_grad()
    def t2i_evaluate(self, subset_name, dataloader):
        tqdm_t2i_loader = tqdm(dataloader)
        tqdm_t2i_loader.set_description(f"Text to Image retrieval on <{subset_name}>")
        t2i_scores = []
        t2i_correct_num = 0
        total_num = 0
        for batch in tqdm_t2i_loader:
            bs = len(batch['label'])
            # get query texts
            query_texts = batch['query_text']  # [B x STR]

            # compute normalized text embeddings
            text_input = self.tokenizer(text=query_texts, return_tensors="pt", padding="max_length", max_length=77).to(self.device)
            text_embeddings = self.model.flava.get_text_features(**text_input).cpu().numpy()[:, 0, :]
            text_embeddings = text_embeddings / np.linalg.norm(text_embeddings, axis=1, keepdims=True)
            text_embeddings = torch.tensor(text_embeddings)  # B,D (D:feature dim)

            # get candidate images
            candidate_images = batch['candidate_images']  # [BxL, PIL.Image]

            # compute normalized image embeddings
            image_embeddings = []
            for images in candidate_images:
                image_inputs = self.feature_extractor(images=images, return_tensors="pt").to(self.device)
                image_embed = self.model.flava.get_image_features(**image_inputs).cpu().numpy()[:, 0, :]
                image_embed = image_embed / np.linalg.norm(image_embed, axis=1, keepdims=True)
                image_embed = torch.tensor(image_embed)   # (L, D)
                image_embeddings.append(image_embed)
            image_embeddings = torch.stack(image_embeddings, dim=0)    # (B, L, D)

            # calculate matching result
            batch_t2i_scores = torch.einsum('BD,BKD->BK', [text_embeddings, image_embeddings]).cpu()
            t2i_scores.append(batch_t2i_scores)
            gt_labels = batch['label']
            pred_labels = batch_t2i_scores.argmax(dim=-1)
            correct_num = (gt_labels == pred_labels).sum()
            t2i_correct_num += correct_num.item()
            total_num += bs

        t2i_scores = torch.cat(t2i_scores, dim=0)
        t2i_acc = 100 * t2i_correct_num / total_num

        return t2i_scores, t2i_acc
