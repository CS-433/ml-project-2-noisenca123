# Using Different PDE Solvers for Noise-NCA-Based Texture Synthesis

This repository contains a code implementation of Using Different PDE Solvers for Noise-NCA-Based Texture Synthesis

## Project Structure

```
.
│  DPS_NCA.ipynb # Well-organized and documented code 
│  README.md
│  requirements.txt # The packages you need to import 
│
├─model # Pre-trained models
│      model_euler_1.pth
│      model_euler_10.pth
│      model_euler_11.pth
│      model_euler_12.pth
│      model_euler_13.pth
│      model_euler_14.pth
│      model_euler_15.pth
│      model_euler_16.pth
│      model_euler_17.pth
│      model_euler_18.pth
│      model_euler_19.pth
│      model_euler_2.pth
│      model_euler_20.pth
│      model_euler_3.pth
│      model_euler_4.pth
│      model_euler_5.pth
│      model_euler_6.pth
│      model_euler_7.pth
│      model_euler_8.pth
│      model_euler_9.pth
│      model_im_euler_1.pth
│      model_im_euler_10.pth
│      model_im_euler_11.pth
│      model_im_euler_12.pth
│      model_im_euler_13.pth
│      model_im_euler_14.pth
│      model_im_euler_15.pth
│      model_im_euler_16.pth
│      model_im_euler_17.pth
│      model_im_euler_18.pth
│      model_im_euler_19.pth
│      model_im_euler_2.pth
│      model_im_euler_20.pth
│      model_im_euler_3.pth
│      model_im_euler_4.pth
│      model_im_euler_5.pth
│      model_im_euler_6.pth
│      model_im_euler_7.pth
│      model_im_euler_8.pth
│      model_im_euler_9.pth
│      model_rk4_1.pth
│      model_rk4_10.pth
│      model_rk4_11.pth
│      model_rk4_12.pth
│      model_rk4_13.pth
│      model_rk4_14.pth
│      model_rk4_15.pth
│      model_rk4_16.pth
│      model_rk4_17.pth
│      model_rk4_18.pth
│      model_rk4_19.pth
│      model_rk4_2.pth
│      model_rk4_20.pth
│      model_rk4_3.pth
│      model_rk4_4.pth
│      model_rk4_5.pth
│      model_rk4_6.pth
│      model_rk4_7.pth
│      model_rk4_8.pth
│      model_rk4_9.pth
│
└─result # Empty folder for saving output results, may not in the repository

```

## Implementation of DPS_NCA.ipynb

Our code is presented in a .ipy file instead of multiple .py files because the .ipynb file provides better and more unified visualization. 
The `DPS_NCA.ipynb` file is still well-organized, consisting of these main following code blocks:

- Load dataset: This section loads the dataset for you.
- Imports and Notebook Utilities: This section mainly defines some functions and classes for visualizing texture videos.
- Define Style Loss: This section defines the loss function. In our experiment, we use the RelaxedOT Loss to remain consistent with previous work.
- NoiseNCA Architecture: This section defines the structure of the network and the different solvers.
- Training and Saving Models: Train the model on 20 images using three different solvers and save the results.
- Testing Models: Test using different solver combinations and output the results to the `result` folder.
- Testing Results Visualization: Visualize the results, and you can view the generated videos in the current directory. When modifying the model used, 
you may need to adjust the `image_path` to ensure the proper calculation of the loss function (if you only want to view the visualization results, you can ignore this).
- Wilcoxoon Testing: Perform Wilcoxon Testing.

Each code block contains detailed and accurate comments, allowing you to understand the implementation of the method and the configuration of the parameters.
## Setup and Usage

1. Clone this repository
2. Ensure that you have installed the CUDA drivers and the PyTorch version compatible with CUDA; you also need to make sure that FFmpeg is installed. 
The official download website is [https://ffmpeg.org/download.html](https://ffmpeg.org/download.html).
3. Now, you just need to create a virtual environment and install the required packages. Run the following command:
- python -m venv env
- source env/bin/activate, For Windows: .\env\Scripts\activate
- pip3 install -r requirements.txt

The CUDA-compatible version of PyTorch is essential. If the correct version is not installed, reinstall it using the following command:

- pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

4. Now, start Jupyter Notebook in the virtual environment and run each module of `DPS_NCA.ipynb` sequentially. You can check our experimental
 results. By using the pretrained models in the `model` folder, you can skip the training module.

## Dependencies

The implementation uses these libraries:

Python built-in libraries:
- os
- zipfile
- io
- base64
- json
- glob
- warnings

Other libraries:
- numpy
- gdown
- PIL.Image
- PIL.ImageDraw
- requests
- matplotlib.pyplot
- IPython.display
- tqdm.notebook
- moviepy
- torch
- torchvision
- scipy.stats

## Authors

- Xinran Li
- Siyuan Zhang

**_Authorship is alphabetical and does not reflect individual contributions._**

Machine Learning Course (CS-433), Fall 2024 <br>
School of Computer and Communication Sciences (IC) <br>
EPFL (École polytechnique fédérale de Lausanne) <br>
Switzerland
