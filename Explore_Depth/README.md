# Exploring the Role of Depth Information in RS Tasks

## Installation
To set up the environment, please follow these steps:

1. Create and activate a conda environment :
    ```shell
    conda create -n depth_env python=3.7
    conda activate depth_env
    ```

2. Install the required packages:
    ```shell
    pip install -r requirements.txt
    ```

## Datasets
We will provide access to the data used for training. As these processed datasets will soon be available for download via Baidu Netdisk.


## Usage


1.Modules for fully supervised, semi-supervised, and unsupervised domain adaptation tasks are provided in `method.py`, which can be easily migrated to your task. In addition, we provide examples for each method on how to use these modules. These examples are implemented on fully supervised networks, WSCL, and FT_GLGAN, respectively.

2.The depth estimation image, essential for auxiliary training, can be obtained by utilizing Depth Anything v2 for processing the image.
