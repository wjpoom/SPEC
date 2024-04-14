# Evaluate Custom Vision Language Model on SPEC
We have implemented the testing code for four VLMs, namely [CLIP](https://arxiv.org/abs/2103.00020), [BLIP](https://arxiv.org/abs/2201.12086), [FLAVA](https://arxiv.org/abs/2112.04482), and [CoCa](https://arxiv.org/abs/2205.01917). 
If you want to test other custom models, you need to complete the following steps.

## Step 1. Implement custom `model wrapper`
Firstly, create a file named `custom_wrapper.py` under [`spec/models/`](https://github.com/wjpoom/SPEC/tree/main/spec/models). Next, define your own `CustomWrapper` class within this file. 
`CustomWrapper` needs to inherit from the base class [`BaseWrapper`](https://github.com/wjpoom/SPEC/blob/d1048b57b4f64a813624ce6575ececa86a9178ea/spec/models/base_wrapper.py#L6) 
and implement the `i2t_evaluate` and `t2i_evaluate` methods, you can also add any other methods you need. Your `CustomWrapper` should look like the following:
```python
class CustomWrapper(BaseWrapper):
    def __init__(self):
        pass
    @torch.no_grad()
    def i2t_evaluate(self, subset_name, dataloader):
        pass
    @torch.no_grad()
    def t2i_evaluate(self, subset_name, dataloader):
        pass
```
**Note**: take care of the return format of `i2t_evaluate` and `t2i_evaluate`. Please refer to instances in [`CLIPWrapper`](https://github.com/wjpoom/SPEC/blob/d1048b57b4f64a813624ce6575ececa86a9178ea/spec/models/clip_wrapper.py#L9C7-L9C18), [`BLIPWrapper`](https://github.com/wjpoom/SPEC/blob/d1048b57b4f64a813624ce6575ececa86a9178ea/spec/models/blip_wrapper.py#L30) or
[`FLAVAWrapper`](https://github.com/wjpoom/SPEC/blob/d1048b57b4f64a813624ce6575ececa86a9178ea/spec/models/flava_wrapper.py#L8) when implementing your code.

## Step 2. Add your model in `get_model()` 
We defined a method named `get_model()` in [`spec/models/__init__.py`](https://github.com/wjpoom/SPEC/blob/main/spec/models/__init__.py),
which handles model loading. You need to add the code to load your custom model within this function,
simply add the following code block at the end:
```python
elif model_name == CUSTOM_MODEL_NAME:
    from .custom_wrapper import CUSTOMWrapper
    model = CUSTOMWrapper(...)
    image_preprocess = model.image_preprocess
    return model, image_preprocess
```
where `CUSTOM_MODEL_NAME` is a string that distinguishes your custom model, `custom_wrapper` and `CUSTOMWrapper` are the filename and wrapper class name you defined in the first step.

**Note**: You need to return `image_preprocess`, which will be used in the dataset construction to process the input image (e.g., cropping, converting to tensor, etc.). If you don't need this operation, please return None.
## Step 3. Evaluate your custom model
Run the following script to evaluate your custom model on SPEC !
```shell
model=CUSTOM_MODEL_NAME
model_dir='/path/to/cache/models'
data_dir='/path/to/data'
out_dir='/path/to/save/results'

python eval.py \
--model-name $model \
--model-cache-dir $model_dir \
--subset-names absolute_size relative_size absolute_spatial relative_spatial existence count \
--data-root $data_dir \
--out-path $out_dir \
--batch-size 64 \
--num-workers 8 \
--seed 1 
```
