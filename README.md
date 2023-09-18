# Deepfake Detection Project
* The objective of this project is to develop a deepfake detector. We believe this algorithmic approach has the potential to effectively detect manipulated or digitally generated images, with significant implications across diverse fields, including criminal science. However, this task is not as simple as seen. Due to the high complexity of the project, we have decided to use deep learning models throughout the project rather than machine learning models. To achieve our goal, we used a dataset called [“deepfake_faces”](https://www.kaggle.com/datasets/dagnelies/deepfake-faces) from Kaggle that contains both real images cropped from videos and computer-generated fake images.
* Preprocessing of the dataset was necessary to deal with the class imbalance and model resolution compatibilities. Techniques such as undersampling, oversampling, and augmentation have been tested.
* Transfer learning has been used through fine-tuning pre-trained models with architectures including Nvidia Efficient Net B4 and B7, Facebook Deit Tiny, ResNet-50, and InceptionResNet-50.
* These fine-tuned models have also been compared with scratch models we built from the ground up.
## Contributors
- [Alperen Gözeten](https://github.com/alperengozeten)
- [Korhan Kemal Kaya](https://github.com/korhankemalkaya)
- [Oğuz Can Duran](https://github.com/oguzcanduran)
- [Can Ersoy](https://github.com/CanErsoy20)
