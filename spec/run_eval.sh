model_dir='/path/to/cache/models'
data_dir='/path/to/data'
out_dir='/path/to/save/results'

for model in clip blip flava coca
do
    python eval.py \
    --model-name $model \
    --model-cache-dir $model_dir \
    --subset-names absolute_size relative_size absolute_spatial relative_spatial existence count \
    --data-root $data_dir \
    --out-path $out_dir \
    --batch-size 64 \
    --num-workers 8 \
    --seed 1 
done