# OpenGesture 0.0.1v
Platform | Build Status |
-------- | ------------ |
Azure | [![Build status](https://ci.appveyor.com/api/projects/status/swutsp1bjcc56q64/branch/master?svg=true)](https://ci.appveyor.com/project/ddiakopoulos/hand-tracking-samples/branch/master)


## Environment Setup
* Install [OpenVINO™ Toolkit](https://software.intel.com/en-us/openvino-toolkit) ([guide](https://software.intel.com/en-us/articles/OpenVINO-Install-Linux))<br>
    	**Note**: Please use  *root privileges* to run the installer when installing the core components.
* Install OpenCL Driver for GPU
	```bash
	cd /opt/intel/computer_vision_sdk/install_dependencies
	sudo ./install_NEO_OCL_driver.sh
	```
* Install Intel® RealSense™ SDK 2.0 [(tag v2.14.1)](https://github.com/IntelRealSense/librealsense/tree/v2.14.1)<br>
	* [Install from source code](https://github.com/IntelRealSense/librealsense/blob/v2.14.1/doc/installation.md)(Recommended)<br>
	* [Install from package](https://github.com/IntelRealSense/librealsense/blob/v2.14.1/doc/distribution_linux.md)<br>
* [Build OpenCV from Source](https://docs.opencv.org/trunk/d7/d9f/tutorial_linux_install.html):Please use git checkout 3.4 to use version 3.4
* [Install Qt5 for Ubuntu](https://wiki.qt.io/Install_Qt_5_on_Ubuntu)


Note that the inference engine backend is used by default since OpenCV 3.4.2 (OpenVINO 2018.R2) when OpenCV is built with the Inference engine support, so the call above is not necessary. Also, the Inference engine backend is the only available option (also enabled by default) when the loaded model is represented in OpenVINO™ Model Optimizer format.
      

# 1. Research previous and current work on Hand Gestures



## 2. Defining hand gestures.
Horns Sign                 |  Peace Sign
:-------------------------:|:-------------------------:
![](https://raw.githubusercontent.com/TebogoNakampe/OpenGesture/master/assets/a4g.png)  |  ![](https://raw.githubusercontent.com/TebogoNakampe/OpenGesture/master/assets/peace.png)



