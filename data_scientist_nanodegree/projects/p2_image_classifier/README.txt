NOTE: need training/validation/test data to run as is with the 'recommended usage' commands below, but was not sure how to download entire workspace to include the files, and not sure I should include all of those files in my submission.

NOTE: usage is slightly different than specs in https://classroom.udacity.com/nanodegrees/nd025/parts/55eca560-1498-4446-8ab5-65c52c660d0d/modules/627e46ca-85de-4830-b2d6-5471241d5526/lessons/c3578df2-1f0d-4182-9b6c-b63a96728d7c/concepts/4db448a2-9b6c-4df8-9685-d905814bca04, for consistency reasons,
but the rubric does not specify specific usage.

TRAIN:

```
python train.py --data_dir='../aipnd-project/flowers' --save_dir='/' --device='cuda' --epochs=10 --learning_rate=0.001 --hidden_units1=12544 --hidden_units2=6272 --drop1=0.5 --drop2=None --arch='vgg16
```


PREDICT:

```
python predict.py --json_map='cat_to_name.json' --device='cuda' --save_path='checkpoint.pth' --image_path=None --arch='vgg16' --top_k=5 
```

RECOMMENDED USAGE:

python train.py --data_dir='../aipnd-project/flowers' --epochs=1
python predict.py --image_path='../aipnd-project/flowers/test/1/image_06764.jpg'
