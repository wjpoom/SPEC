import torch

from einops import rearrange
from tqdm import tqdm
from .base_wrapper import BaseWrapper
from open_clip import create_model_and_transforms, get_tokenizer


class CLIPWrapper(BaseWrapper):
    def __init__(self, device, variant='ViT-B-32', pretrained='openai'):
        self.device = device
        model, _, image_preprocess = create_model_and_transforms(variant,
                                                                 device=self.device,
                                                                 pretrained=pretrained)
        self.model = model.eval()
        self.tokenizer = get_tokenizer(variant)
        self.image_preprocess = image_preprocess

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
            query_images = batch['query_image'].to(self.device) # B,C,H,W (B:batch size)

            # compute normalized image embeddings
            image_embeddings = self.model.encode_image(query_images)  # B,D (D:feature dim)
            image_embeddings /= image_embeddings.norm(dim=-1, keepdim=True)

            # get candidate texts
            candidate_texts = batch['candidate_texts']
            candidate_texts = [self.tokenizer(texts) for texts in candidate_texts]
            candidate_texts = torch.stack(candidate_texts, dim=0).to(self.device)  # B,L,S (L:num of candidate texts, S:sentence length)

            # compute normalized text embeddings
            candidate_texts = rearrange(candidate_texts, 'B L S -> (B L) S')
            text_embeddings = self.model.encode_text(candidate_texts)  # BL,D (D:feature dim)
            text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)
            text_embeddings = rearrange(text_embeddings, '(B L) D -> B L D', B=bs) # B, L, D

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
            query_texts = batch['query_text']  # B,1,S (B:batch size, S:sentence length)
            query_texts = self.tokenizer(query_texts).to(self.device)  # B, S (B:batch size, S:sentence length)

            # compute normalized text embeddings
            text_embeddings = self.model.encode_text(query_texts)  # B,D (D:feature dim)
            text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)

            # get candidate images
            candidate_images = batch['candidate_images'].to(self.device)  # B,K,C,H,W (K:num of candidate images per case, S:sentence length)
            candidate_images = rearrange(candidate_images, 'B K C H W -> (B K) C H W')

            # compute normalized image embeddings
            image_embeddings = self.model.encode_image(
                candidate_images)  # BK,D (K:num of candidate images per case, D:feature dim)
            image_embeddings /= image_embeddings.norm(dim=-1, keepdim=True)
            image_embeddings = rearrange(image_embeddings, '(B K) D -> B K D', B=bs)

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
