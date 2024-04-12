import os
import torch
import yaml
import subprocess
import torch.nn.functional as F

from tqdm import tqdm
from einops import rearrange
from .blip_utils.blip_retrieval import blip_retrieval
from .base_wrapper import BaseWrapper
from torchvision import transforms


# All the below URLs are taken from, and most of the implementation are heavily inspired from the wonderful https://github.com/salesforce/BLIP repo.
download_urls = {
    "blip-flickr-base": {
        "model_url": "https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_retrieval_flickr.pth",
        "config_url": "https://raw.githubusercontent.com/salesforce/BLIP/0480d94d5725a3d1aac66f21e6bf138ac17d323d/configs/retrieval_flickr.yaml",
        "bert_config_url": "https://raw.githubusercontent.com/salesforce/BLIP/0480d94d5725a3d1aac66f21e6bf138ac17d323d/configs/med_config.json"
    },

    "blip-coco-base": {
        "model_url": "https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_retrieval_coco.pth",
        "config_url": "https://github.com/salesforce/BLIP/raw/0480d94d5725a3d1aac66f21e6bf138ac17d323d/configs/retrieval_coco.yaml",
        "bert_config_url": "https://raw.githubusercontent.com/salesforce/BLIP/0480d94d5725a3d1aac66f21e6bf138ac17d323d/configs/med_config.json"
    },
}


class BLIPModelWrapper(BaseWrapper):
    def __init__(self, cache_dir, device, variant="blip-coco-base"):
        self.cache_dir = cache_dir
        self.device = device
        self.variant = variant
        self.image_preprocess = transforms.Compose([
                        transforms.Resize((384, 384), interpolation=transforms.functional.InterpolationMode.BICUBIC),
                        transforms.ToTensor(),
                        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])
        # download and load model
        self.config_path = os.path.join(cache_dir, f"{variant}-config")
        self.model_path = os.path.join(cache_dir, f"{variant}.pth")
        self.bert_config_path = os.path.join(cache_dir, "configs", f"{variant}_med_config.json")
        if not (os.path.exists(self.config_path) and os.path.exists(self.model_path) and os.path.exists(
                self.bert_config_path)):
            self.download()
        config = yaml.load(open(self.config_path, 'r'), Loader=yaml.Loader)
        self.config = config
        self.config['k_test'] = 128
        config['med_config'] = self.bert_config_path
        model = blip_retrieval(pretrained=self.model_path, image_size=config['image_size'], vit=config['vit'],
                               vit_grad_ckpt=config['vit_grad_ckpt'], vit_ckpt_layer=config['vit_ckpt_layer'],
                               queue_size=config['queue_size'], negative_all_rank=config['negative_all_rank'],
                               med_config=config['med_config']).to(device)
        self.model = model.eval()


    def download(self):
        print(f"Downloading BLIP model to {self.cache_dir}...")
        model_url = download_urls[self.variant]["model_url"]
        config_url = download_urls[self.variant]["config_url"]
        bert_config_url = download_urls[self.variant]["bert_config_url"]
        os.makedirs(os.path.join(self.cache_dir, "configs"), exist_ok=True)
        subprocess.call(["wget", "-cq", model_url, "-O", self.model_path])
        subprocess.call(["wget", "-cq", config_url, "-O", self.config_path])
        subprocess.call(["wget", "-cq", bert_config_url, "-O", self.bert_config_path])

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
            query_images = batch['query_image'].to(self.device)  # B,C,H,W (B:batch size)

            # compute normalized image embeddings
            image_feats = self.model.visual_encoder(query_images)
            image_embeddings = self.model.vision_proj(image_feats[:, 0, :])
            image_embeddings = F.normalize(image_embeddings, dim=-1)  # B, D (D:feature dim)

            # get candidate texts
            candidate_texts = batch['candidate_texts']

            # compute normalized text embeddings
            text_embeddings = []
            for texts in candidate_texts:
                text_input = self.model.tokenizer(texts, padding='max_length', truncation=True, max_length=35, return_tensors="pt").to(self.device)
                text_feat = self.model.text_encoder(text_input.input_ids, attention_mask=text_input.attention_mask, mode='text')
                text_embed = F.normalize(self.model.text_proj(text_feat.last_hidden_state[:, 0, :]))
                text_embeddings.append(text_embed)
            text_embeddings = torch.stack(text_embeddings, dim=0)   # B, L, D

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
            query_texts = batch['query_text']  # B (B:batch size, list)
            query_texts = self.model.tokenizer(query_texts, padding='max_length', truncation=True, max_length=35, return_tensors="pt").to(self.device)

            # compute normalized text embeddings
            text_feats = self.model.text_encoder(query_texts.input_ids, attention_mask=query_texts.attention_mask, mode='text')
            text_embeddings = F.normalize(self.model.text_proj(text_feats.last_hidden_state[:, 0, :]))  # B,D (D:feature dim)

            # get candidate images
            candidate_images = batch['candidate_images'].to(self.device)  # B,K,C,H,W (K:num of candidate images per case, S:sentence length)
            candidate_images = rearrange(candidate_images, 'B K C H W -> (B K) C H W')

            # compute normalized image embeddings
            image_feats = self.model.visual_encoder(candidate_images)
            image_embeddings = self.model.vision_proj(image_feats[:, 0, :])
            image_embeddings = F.normalize(image_embeddings, dim=-1)  # B， D (D:feature dim)
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
