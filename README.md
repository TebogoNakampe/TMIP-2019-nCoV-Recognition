# Treatise of Medical Image Processing (TMIP) v0.2.0
# Volume 2: 
            Coronavirus (2019-nCoV infection) Recognition using Deep Neural Networks for Computer Tomography (CT) image analysis.

<p align="center">
  <img width="460" height="300" src="https://github.com/TebogoNakampe/TMIP-2019-nCoV-Recognition/blob/master/Coronavirus-CDC-645x645.jpg">
</p>

In this project, I built a model to classify brain tumours into three types based on MRI scans: Astrocytoma, Oligodendroglioma or Glioblastoma.

From a dataset of 32 patients, tumour features such as size, enhancement quality, necrosis proportion, etc. were extracted by radiologists. Diagnosis was also established for these patients. Based on this information I was able to create an optimised model to classify tumours with a 90% cross-validated accuracy.

# Data Preprocessing and Model Selection:
                                          Data_preprocessing.ipynb
                                          TMIP_BrainTumour.ipynb
                                          Data = REMBRANDT may be found at (https://wiki.cancerimagingarchive.net/display/Public/REMBRANDT#4b0fe4760f6d405e9d09ad75c6f54790)
                                                                                                         
<p align="center">
  <img width="460" height="300" src="https://github.com/TebogoNakampe/Treatise-of-Medical-Image-Processing/blob/master/output_75_1.png">
</p>
The main approach was to extract tumour features from 120 patients MRI scans for whom diagnosis was established by neurologis. A detailed notebook containing an exploration of several image processing methods can be found in TMIP.ipynb

TMIP using FCN and Simple CNN:
In order apply this model to new patients and generate an unbiased estimate of the model's performance, we are exploring simple convolutional neural networks and Fully Convolutional Neural Networks. However while we did manage to extract features such as tumour dimensions, side of epicentre, T1/FLAIR ratio and Enhancement Quality, so far we have been unable to extract features such as necrosis proportion or thickness of enhancing margin. 

Data Preprocessing then FCN and simple CNN:
               Simplified_FCN.ipynb
                medical-image-segmentation.ipynb
                


# Requirements for TMIP to work: 

            Linux
            Intel MKL-DNN
            Tensorflow version 1.1.0
 # Installing Dependencies (Anaconda installation is recommended)

    pip install scipy
    pip install imageio
    pip install pyssim
    pip install joblib
    pip install Pillow
    pip install scikit-image
    pip install opencv-python
    pip install pytube
    sudo apt-get install unrar

FFMPEG needs to be installed:

    conda install -c menpo ffmpeg=3.1.
    Get NIPYPE: 
    https://github.com/nipy/nipype
    Get FSL:
    https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FslInstallation
    
    
# Citation

If you find this useful, please cite our work as follows:

@article{tebogonakampe17TMIP,
  author = {Tebogo Nakampe and Thabo Koee,
  title = {Treatise of Medical Image Processing},
  journal = {TMIP},
  year = {2017},
}

Please contact "afribizintegration@gmail.com" if you have any questions.


