Trained ResNet Torch models
============================

These are ResNet models trainined on ImageNet. The accuracy on the ImageNet validation set are included below.

- [ResNet-18](http://torch7.s3-website-us-east-1.amazonaws.com/data/resnet-18.t7)
- [ResNet-34](http://torch7.s3-website-us-east-1.amazonaws.com/data/resnet-34.t7)
- [ResNet-50](http://torch7.s3-website-us-east-1.amazonaws.com/data/resnet-50.t7)
- [ResNet-101](http://torch7.s3-website-us-east-1.amazonaws.com/data/resnet-101.t7)

The ResNet-50 model has a batch normalization layer after the addition, instead of immediately after the convolution layer.

##### ImageNet 1-crop error rates

| Network       | Top-1 error | Top-5 error |
| ------------- | ----------- | ----------- |
| ResNet-18     | 30.43       | 10.76       |
| ResNet-34     | 26.73       | 8.74        |
| ResNet-50     | 24.01       | 7.02        |
| ResNet-101    | **22.44**   | **6.21**    |

##### ImageNet 10-crop error rates

| Network       | Top-1 error | Top-5 error |
| ------------- | ----------- | ----------- |
| ResNet-18     | 28.22       | 9.42        |
| ResNet-34     | 24.76       | 7.35        |
| ResNet-50     | 22.24       | 6.08        |
| ResNet-101    | **21.08**   | **5.35**    |


### Fine-tuning on a custom dataset

Your images don't need to be pre-processed or packaged in a database, but you need to arrange them so that your dataset contains a `train` and a `val` directory, which each contain sub-directories for every label. For example:

```
train/<label1>/<image.jpg>
train/<label2>/<image.jpg>
val/<label1>/<image.jpg>
val/<label2>/<image.jpg>
```

You can then use the included [ImageNet data loader](datasets/imagenet.lua) with your dataset and train with the `-resetClassifer` and `-nClasses` options:

```bash
th main.lua -retrain resnet-50.t7 -data [path-to-directory-with-train-and-val] -resetClassifier true -nClasses 80
```

The labels will be sorted alphabetically. The first output of the network corresponds to the label that comes first alphabetically.

### Extracting image features

The [`extract-features.lua`](extract-features.lua) script will extract the image features from an image and save them as a serialized Torch tensor. To use it, first download one of the trained models above. Next run it using

```bash
th extract-features.lua resnet-101.t7 img1.jpg img2.jpg ...
```

This will save a file called `features.t7` in the current directory. You can then load the image features in Torch.

```lua
local features = torch.load('features.t7')
```
