"""code credict: https://github.com/mertyg/vision-language-models-are-bows/tree/main/model_zoo"""

def get_model(model_name, cache_dir='~/.cache', device='cuda'):
    """
    Helper function that returns a model and an image preprocessing function and text tokenizer.
    Args:
        model_name: the model that you want to create
        cache_dir: the path to cache the downloader model checkpoints
        device
    Returns:
        pretrained_model, image_preprocess
    """

    if model_name == 'clip':
        from .clip_wrapper import CLIPWrapper
        clip_model = CLIPWrapper(device=device, variant='ViT-B-32', pretrained='openai')
        image_preprocess = clip_model.image_preprocess
        return clip_model, image_preprocess

    elif model_name == 'blip':
        from .blip_wrapper import BLIPModelWrapper
        blip_model = BLIPModelWrapper(cache_dir=cache_dir, device=device, variant="blip-coco-base")
        image_preprocess = blip_model.image_preprocess
        return blip_model, image_preprocess
    
    elif model_name == "flava":
        from .flava_wrapper import FlavaWrapper
        flava_model = FlavaWrapper(cache_dir=cache_dir, device=device)
        image_preprocess = None
        return flava_model, image_preprocess

    elif model_name == "coca":
        from .clip_wrapper import CLIPWrapper
        coca_model = CLIPWrapper(device=device, variant="coca_ViT-B-32", pretrained="laion2B-s13B-b90k")
        image_preprocess = coca_model.image_preprocess
        return coca_model, image_preprocess

    else:
        raise ValueError(f"Unknown model {model_name}")
