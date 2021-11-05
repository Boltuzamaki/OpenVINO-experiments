### Convert VGG16 

```
python mo_tf.py --saved_model_dir "./experiments/model" --output_dir "./experiments/new" --input_shape=(1,224,224,3)
```

### Convert Inception v3

```
python mo_tf.py --saved_model_dir "./experiments/model" --output_dir "./experiments/new" --input_shape=(1,224,224,3) --mean_value=[127.5,127.5,127.5] --scale=127.5
```