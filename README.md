# Treatise of Medical Image Processing (TMIP) v0.2.0
# Volume 2: 
            Coronavirus (2019-nCoV infection) Recognition using Deep Neural Networks for Computer Tomography (CT) image analysis.

<p align="center">
  <img width="460" height="300" src="https://github.com/TebogoNakampe/TMIP-2019-nCoV-Recognition/blob/master/mini_blob.jpeg">
</p>

On Dec. 31, 2019, the World Health Organization (WHO) learned of several cases of a respiratory 
illness clinically resembling viral pneumonia and manifesting as fever, cough, and shortness of 
breath. The newly discovered virus emerging from Wuhan City, Hubei Province of China, was 
temporarily named “novel coronavirus” (2019-nCoV). It is now known officially as COVID-19. 
This new coronavirus belongs to a family of viruses that include Severe Acute Respiratory 
Syndrome (SARS) and Middle East Respiratory Syndrome (MERS). 
The outbreak is escalating quickly, with hundreds of thousands of confirmed COVID-19 cases 
reported globally. Early disease recognition is critical not only for prompt treatment, but also for 
patient isolation and effective public health containment and response. Thus we propose the 
use of AI based CT image analysis for recognition of coronavirus infection under Project 
Treatise of Medical Image Processing v0.2.0.. 
There are a limited number of COVID-19 test kits available in hospitals due to the increasing 
cases daily. Therefore, it is necessary to implement an automatic detection system as a quick 
alternative diagnosis option to prevent COVID-19 spreading among people. Thus we propose 
the use of Deep Neural Networks, as an initial experiment we used ChexNeXt Pneumonia 
Detection Model as the baseline architecture where we use transfer learning to detect 
pneumonia. Secondly we use three different convolutional neural network architectures 
(ResNet50, InceptionV3 and Inception-ResNetV2) for the detection of coronavirus pneumonia 
infected patients using chest X-ray radiographs. 

# Data Preprocessing and Model Selection:
                                          tmipv020.ipynb
                                          Data = ....TBA
                                                                                                         
<p align="center">
  <img width="460" height="300" src="https://github.com/TebogoNakampe/TMIP-2019-nCoV-Recognition/blob/master/Coronavirus-CDC-645x645.jpg">
</p>
Machine Learning Model refers to a mathematical algorithm that is able to find hidden patterns 
based on large number data, in this case the data is X-Ray / CT Scan imagery.  
The model is trained on the viral pneumonia (COVID-19) dataset, one of the largest public 
repositories of COVID-19 radiographs, containing about 200 frontal-view chest radiographs of 
157 unique patients. Each image in the dataset was labeled by radiologists from different 
hospitals where patients infected with COVID-19 were diagnosed. 
Model training process consists of 2 consecutive stages to account for the partially incorrect 
labels in the COVID-19 dataset. First, an ensemble of networks is trained on the training set to 
predict the probability that each of the 14 pathologies is present in the image. The predictions of 
this ensemble are used to relabel the training and tuning sets. A new ensemble of networks are 
finally trained on this relabeled training set. 
Without any additional supervision, the model produces heat maps that identify locations in the 
chest radiograph that classify COVID-19 among other pathologies. 
We thus serve the model to scientists and medical professionals through an Inference Cloud 
application powered by Intel OpenVINO Toolkit. For faster media processing we empoy Intel 
Open Visual Cloud architecture optimized for Intel Xeon Scalable processors. 
                


# Requirements for TMIP to work: 

            Linux
            Intel MKL-DNN
            Tensorflow version 1.1.0
 # Installing Dependencies (Anaconda installation is recommended)

            absl-py==0.9.0
            args==0.1.0
            astor==0.8.1
            cachetools==4.0.0
            certifi==2019.11.28
            chardet==3.0.4
            cycler==0.10.0
            gast==0.2.2
            google-auth==1.11.3
            google-auth-oauthlib==0.4.1
            google-pasta==0.2.0
            grpcio==1.27.2
            h5py==2.10.0
            idna==2.9
            imutils==0.5.3
            joblib==0.14.1
            Keras==2.3.1
            Keras-Applications==1.0.8
            Keras-Preprocessing==1.1.0
            kiwisolver==1.1.0
            Markdown==3.2.1
            matplotlib==3.2.0
            numpy==1.18.1
            oauthlib==3.1.0
            opencv-python==4.2.0.32
            opt-einsum==3.2.0
            Pillow==7.0.0
            protobuf==3.11.3
            pyasn1==0.4.8
            pyasn1-modules==0.2.8
            pyparsing==2.4.6
            python-dateutil==2.8.1
            PyYAML==5.3
            requests==2.23.0
            requests-oauthlib==1.3.0
            rsa==4.0
            scikit-learn==0.22.2.post1
            scipy==1.4.1
            six==1.14.0
            tensorboard==2.1.1
            tensorflow==2.1.0
            tensorflow-cpu==2.1.0
            tensorflow-estimator==2.1.0
            termcolor==1.1.0
            urllib3==1.25.8
            Werkzeug==1.0.0

FFMPEG needs to be installed:

    conda install -c menpo ffmpeg=3.1.
    Get NIPYPE: 
    https://github.com/nipy/nipype
    Get FSL:
    https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FslInstallation
    
    
# Citation

If you find this useful, please cite our work as follows:

@article{tebogonakampe17TMIP,
  author = {Tebogo Nakampe,
  title = {Treatise of Medical Image Processing v020},
  journal = {TMIPv020},
  year = {2020},
}

Please contact "info@4ir-abi.co.za" if you have any questions.


