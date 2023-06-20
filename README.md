<div align="center">
 
![logo](https://github.com/souvikmajumder26/Land-Cover-Semantic-Segmentation-PyTorch/blob/main/assets/logo2.jpg)  

<h1 align="center"><strong>🛣 Land-Cover-Semantic-Segmentation-PyTorch:<h6 align="center">An end-to-end Image Segmentation (CV) project</h6></strong></h1>

![PyTorch - Version](https://img.shields.io/badge/PYTORCH-2.0+-red?style=for-the-badge&logo=pytorch)
![Python - Version](https://img.shields.io/badge/PYTHON-3.9+-blue?style=for-the-badge&logo=python&logoColor=white)
[![Generic badge](https://img.shields.io/badge/License-MIT-<COLOR>.svg?style=for-the-badge)](https://github.com/souvikmajumder26/Land-Cover-Semantic-Segmentation-PyTorch/blob/main/LICENSE) 
[![GitHub Issues](https://img.shields.io/github/issues/souvikmajumder26/Land-Cover-Semantic-Segmentation-PyTorch.svg?style=for-the-badge)](https://github.com/souvikmajumder26/Land-Cover-Semantic-Segmentation-PyTorch/issues)
![Contributions welcome](https://img.shields.io/badge/contributions-welcome-orange.svg?style=for-the-badge)

</div>

----
 
## 📚 Table of Contents
- [Overview](#overview)
- [Demo](#demo)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Setting up and Running the project with Docker](#with-docker)
  - [Setup without Docker](#setup)
  - [Running the project without Docker](#running-the-project)
- [Citing](#citing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

----

## 📌 Overview <a name="overview"></a>
An end-to-end Computer Vision project focused on the topic of <a href="https://en.wikipedia.org/wiki/Image_segmentation" target="_blank">Image Segmentation</a> (specifically Semantic Segmentation). Although this project has primarily been built with the <a href="https://landcover.ai.linuxpolska.com/" target="_blank">LandCover.ai dataset</a>, the project template can be applied to train a model on any semantic segmentation dataset and extract inference outputs from the model in a <b>promptable</b> fashion. Though this is not even close to actual promptable AI, the term is being used here because of a specific functionality that has been integrated here.

The model can be trained on any or all the classes present in the semantic segmentation dataset with the ability to customize the model architecture, optimizer, learning rate, and a lot more parameters directly from the config file, giving it an <b>exciting AutoML</b> aspect. Thereafter while testing, the user can pass the prompt (in the form of the config variable '<b>test_classes</b>') of the selected classes that the user wants to be present in the masks predicted by the trained model.

For example, suppose the model has been trained on all the 30 classes of the <a href="https://www.cityscapes-dataset.com/" target="_blank">CityScapes dataset</a> and while inferencing, the user only wants the class <b>'parking'</b> to be present in the predicted mask due to a specific use-case application. Therefore, the user can provide the prompt as '<b>test_classes = ['parking']</b>' in the config file and get the desired output.

----

## 💫 Demo <a name="demo"></a>
 <p>
  1. Training the model on <a href="https://landcover.ai.linuxpolska.com/" target="_blank">LandCover.ai dataset</a> with '<b>train_classes</b>': <b>['background', 'building', 'woodland', 'water']</b>...
 </p>
 <p align="center">
  <img width="60%" src="https://github.com/souvikmajumder26/Land-Cover-Semantic-Segmentation-PyTorch/blob/main/assets/training.png">
 </p>
 <p>
  2. Testing the trained model for all the classes used to train the model, i.e. '<b>test_classes</b>': <b>['background', 'building', 'woodland', 'water']</b>...
 </p>
 <p align="center">
  <img width="90%" src="https://github.com/souvikmajumder26/Land-Cover-Semantic-Segmentation-PyTorch/blob/main/assets/all_classes.png">
 </p>
 <p>
  3. Testing the trained model for selective classes as per user input, i.e. '<b>test_classes</b>': <b>['background', 'building', 'water']</b>...
 </p>
 <p align="center">
  <img width="90%" src="https://github.com/souvikmajumder26/Land-Cover-Semantic-Segmentation-PyTorch/blob/main/assets/select_classes.png">
 </p>

---

## 🚀 Getting Started <a name="getting-started"></a>

### ✅ Prerequisites <a name="prerequisites"></a>
 
 - <b>Dataset prerequisite for training</b>:
 
 Before starting to train a model, make sure to download the dataset from <a href="https://landcover.ai.linuxpolska.com/" target="_blank">LandCover.ai</a> or from <a href="https://www.kaggle.com/datasets/adrianboguszewski/landcoverai" target="_blank">kaggle/LandCover.ai</a>, and copy/move over the downloaded directories 'images' and 'masks' to the 'train' directory of the project.

### 🐳 Setting up and Running the project with Docker <a name="with-docker"></a>
 
 First and foremost, make sure that <a href="https://www.docker.com/">Docker</a> is installed and working properly in the system.
 
 > 💡 Check the **Dockerfile** added in the repository. According the instructions provided in the file, comment and uncomment the mentioned lines to setup the docker image and container either to **train** or **test** the model at a time.
 
 1. Clone the repository:
 ```shell
 git clone https://github.com/souvikmajumder26/Land-Cover-Semantic-Segmentation-PyTorch.git
 ```
 2. Change to the project directory:
 ```shell
 cd Land-Cover-Semantic-Segmentation-PyTorch
 ```
 3. Build the image from the Dockerfile:
 ```shell
 docker build -t segment_project_image
 ```
 4. Running the docker image in a docker container:
 ```shell
 docker run --name segment_container segment_project_image
 ```
 5. Copying the output files from the container directory to local project directory after execution is complete:
 ```shell
 docker cp segment_container:/segment_project/models .
 docker cp segment_container:/segment_project/logs .
 docker cp segment_container:/segment_project/output .
 ```
 6. Tidying up:
 ```shell
 docker rm segment_container
 docker rmi segment_project_image
 ```
 
 If <a href="https://www.docker.com/">Docker</a> is not installed in the system, follow the below methods to set up and run the project without Docker.

### 💻 Setup (Without 🐳 Docker) <a name="setup"></a>
 
 1. Clone the repository:
 ```shell
 git clone https://github.com/souvikmajumder26/Land-Cover-Semantic-Segmentation-PyTorch.git
 ```
 2. Change to the project directory:
 ```shell
 cd Land-Cover-Semantic-Segmentation-PyTorch
 ```
 3. Setting up programming environment to run the project:
 
 - If using an installed <a hre="https://docs.conda.io/en/latest/">conda</a> package manager, i.e. either Anaconda or Miniconda, create the conda environment following the steps mentioned below:
 ```shell
 conda create --name <environment-name> python=3.9
 conda activate <environment-name>
 ```
 - If using a directly installed python software, create the virtual environment following the steps mentioned below:
 ```shell
 python -m venv <environment-name>
 <environment-name>\Scripts\activate
 ```
 4. Install the dependencies:
 ```shell
 pip install -r requirements.txt
 ```

### 🤖 Running the project (Without 🐳 Docker) <a name="running-the-project"></a>
 
 Running the model training and testing/inferencing scripts from the project directory. It is not necessary to train the model first mandatorily, as a simple trained model has been provided to run the test and check outputs before trying to fine-tune the model.
 
 1. Run the model training script:
 ```shell
 cd src
 python train.py
 ```
 2. Run the model testing/inferencing script:
 ```shell
 cd src
 python test.py
 ```

----

## 📝 Citing <a name="citing"></a>
```
@misc{Souvik2023,
  Author = {Souvik Majumder},
  Title = {Land Cover Semantic Segmentation PyTorch},
  Year = {2023},
  Publisher = {GitHub},
  Journal = {GitHub repository},
  Howpublished = {\url{https://github.com/souvikmajumder26/Land-Cover-Semantic-Segmentation-PyTorch}}
}
```

----

## 🛡️ License <a name="license"></a>
Project is distributed under [MIT License](https://github.com/souvikmajumder26/Land-Cover-Semantic-Segmentation-PyTorch/blob/main/LICENSE)

---

## 👏 Acknowledgements <a name="acknowledgements"></a>
 - [qubvel/segmentation_models.pytorch](https://github.com/qubvel/segmentation_models.pytorch)
 ```
 @misc{Iakubovskii:2019,
  Author = {Pavel Iakubovskii},
  Title = {Segmentation Models Pytorch},
  Year = {2019},
  Publisher = {GitHub},
  Journal = {GitHub repository},
  Howpublished = {\url{https://github.com/qubvel/segmentation_models.pytorch}}
 }
 ```
 - [LandCover.ai](https://landcover.ai.linuxpolska.com/)
 - [bnsreenu/python_for_microscopists](https://github.com/bnsreenu/python_for_microscopists)
 - [leonardo.ai](https://leonardo.ai)

<p align="right">
 <a href="#top"><b>🔝 Return </b></a>
</p>

---
