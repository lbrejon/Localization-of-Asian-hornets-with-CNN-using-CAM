# Localization-of-Asian-hornets-with-CNN-using-CAM
Detection and localization of Asian hornets using PyTorch

## Table of contents 📝
* [My goals](#my-goals)
* [Technologies](#technologies)
* [Project composition](#project-composition)
* [Description](#description)
* [Sources](#sources)

Estimated reading time : ⏱️ 5min

## My goals 🎯
- Deepen knowledge in Computer Vision with CNNs (ResNet18)
- Learn how to fine-tune and train a model to classify elements (Asian hornets detection)
- Localize elements (Asian hornets) with Class Activation Mapping (CAM)

## Technologies 🖥️
Programming languages:
```bash
- Python (framework PyTorch)
```

## Project composition 📂
```bash
.
├── README.md
│
├── data
│   ├── test
│   │   ├── presence
│   │   │    ├── img1.jpg
│   │   │    ├── img7.jpg
│   │   │    └──  ..
│   │   │
│   │   └── absence
│   │        ├── img2.jpg
│   │        ├── img5.jpg
│   │        └──  ..
│   │
│   └── train
│       │    ├── img3.jpg
│       │    ├── img4.jpg
│       │    └──  ..
│       │
│       └── absence
│            ├── img6.jpg
│            ├── img8.jpg
│            └──  ..
│
├── models
│   └── model.ckpt
│
└── notebooks
    ├── CustomDataSet.py
    └── main.ipynb
```

## Description 📋 
- The Asian hornet is a highly invasive bee predator that arrived in France by boat in a shipment from Asia in the early 2000s. While this invader exterminates gradually the bees of their apiaries, the beekeepers have only few means to to fight them. The objective of this project is to develop a **method to automatically detect an Asian hornet in an image** from a camera installed at the entrance of a hive

- In order to make the **classification phase** as efficient as possible (class 0: "hornet presence", class 1: "hornet absence"), the CNN (pre-trained on ImageNet) was **fine-tuned with the hornet dataset**. An annotated database is available in the ```data/``` folder (download data [here](https://drive.google.com/file/d/1z-O8xSt0Ayo7lUtXl2FyUErJVXJ1sC86/view?usp=sharing)). To fine-tune the model, **the fully connected nodes at the end of the network** (where the actual class label predictions are made) **was replaced with initialized ones:** ```resnet.fc = nn.Linear(resnet.fc.in_features, 2)```. In this project, I seek to classify images with the presence/abscene of an Asian hornet that's why **out_features=2**.

- In my case, I decided to work with a **ResNet18** because of its good performances. During the training phase, the hyper-parameters have been chosen so that they do not induce under-learning or over-learning. Indeed, the training loss gradually dropped (the model learned more efficiently) whereas the validation accuracy increased from 0.7 to almost 0.90.
<p align="center">
  <kbd>
  <img width=600 src="https://user-images.githubusercontent.com/56866008/137592073-bc974af8-44df-45f2-a02f-c28bd001e817.png"><br>
  </kbd><br>
  <b>[Figure 1]: Training loss and validation accuracy graph</b><br>
</p>

- Once the training model was completed, the model spots (predicts) whether or not there was an Asian hornets in the image as shown in Figure 2.
<p align="center">
  <kbd>
  <img width=600 src="https://user-images.githubusercontent.com/56866008/137592692-9029b22e-d203-45ae-91da-ffa9350d2fad.png"><br>
  </kbd><br>
  <b>[Figure 2]: Prediction (absence:0, presence:1)</b><br>
</p>

- Let's focus on **hornets localization in the image using Class Activation Mapping (CAM)**. CNNs have led to impressive performance on a variety of visual recognition tasks and especially the remarkable ability to **localize object** and to identify exactly which regions of an image are being used for discrimination. A class activation map for a particular category indicates **the discriminative image regions used by the CNN to identify that category**.
<p align="center">
  <kbd>
  <img width=600 src="https://user-images.githubusercontent.com/56866008/137592657-c85b9013-4c44-4472-bdd0-0ad6c83b96a9.JPG"><br>
  </kbd><br>
  <b>[Figure 3]: Localization of Asian hornets with CAM</b><br>
</p>

## Sources ⚙️
- Help for Class Activation Mapping [Learning Deep Features for Discriminative Localization](http://cnnlocalization.csail.mit.edu/Zhou_Learning_Deep_Features_CVPR_2016_paper.pdf)
