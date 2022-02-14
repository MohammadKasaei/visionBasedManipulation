# visionBasedManipulation



### Requirements

1. Python 3.6/3.7
2. PyTorch 1.8.0/1.9.0
3. attrdict
4. scikit-image
5. PyBullet

This project is tested on Python 3.6 and 3.7, PyTorch 1.8.0 and 1.9.0.



### Installation guidance

```python
conda create -n <env_name> python=3.7
conda activate env_name
conda install pytorch==1.8.0 torchvision==0.9.0 cudatoolkit=11.1 -c pytorch -c conda-forge # Please make sure that the version of pytorch is compatible with the version of cudatoolkit, and the version of cudatoolkit is compatible with your GPU.
conda install scikit-image
pip install attrdict opencv-python matplotlib
pip install pybullet --upgrade --user
pip install -U rospkg
pip3 install open3d
```



### Environment Setup

If you are using command line, before running our basic demo, you need to add the path to our project to the PYTHONPATH environment variable. Just run the following command.

```shell
export PYTHONPATH=$PYTHONPATH:/your/path/to/visionBasedManipulation:/your/path/to/visionBasedManipulation/network
```

if you are using vscode, you can use the launch.json file in .vscode folder to configure project and  run the demo directly.



###  Basic demo

Activate the virtual environment and run the basic demo.

```python
conda activate env_name
python demos/basicDemo.py
```

You can see a simulation like this: 


![](https://github.com/MohammadKasaei/visionBasedManipulation/blob/main/figs/basicDemo.png)





### Basic demo with ROS wrapper(ongoing)

This demo included a ROS wrapper for the basic demo, the RGB-D images and point clouds are generated from Pybullet simulation and published to ROS topic. 

@TODO:

needs to add tf from Pybullet

```
conda activate env_name
python demos/basicROSDemo.py
```

![](https://github.com/MohammadKasaei/visionBasedManipulation/blob/main/figs/pybullet.png "Pybullet simulation") ![](https://github.com/MohammadKasaei/visionBasedManipulation/blob/main/figs/pybullet-pc.png "Point Cloud from Pybullet")

### Basic VGN demo(ongoing)

This demo adopt [VGN](https://github.com/ethz-asl/vgn.git) in our setup, its a 6D grasp predicting method using point clouds as input.

```
conda activate env_name
python demos/basicVGNDemo.py
```

