# Treatise of Medical Image Processing (TMIP) v0.2.0
Platform | Build Status |
-------- | ------------ |
Azure DSVM | [![Build status](https://ci.appveyor.com/api/projects/status/swutsp1bjcc56q64/branch/master?svg=true)](https://ci.appveyor.com/project/ddiakopoulos/hand-tracking-samples/branch/master)
oneAPI DevCloud| [![Build status](https://ci.appveyor.com/api/projects/status/swutsp1bjcc56q64/branch/master?svg=true)](https://ci.appveyor.com/project/ddiakopoulos/hand-tracking-samples/branch/master)

            Coronavirus (2019-nCoV infection) Recognition using Deep Neural Networks for Computer Tomography (CT) & X-Ray image analysis.

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
                

## Azure Environment Setup
* Get a  [Microsoft Azure Account](https://azure.microsoft.com/en-us/)
* [Create your Data Science Virtual Machine for Linux](https://docs.microsoft.com/en-us/azure/machine-learning/data-science-virtual-machine/dsvm-ubuntu-intro)
* Clone TMIP Repo
	```bash
	git clone https://github.com/TebogoNakampe/TMIP-2019-nCoV-Recognition.git
	cd TMIP-2019-nCoV-Recognition/TMIP_Azure/
	pip install -r requirements.txt
	```
* Get Data and set Path to "rsna-data" and "covid-chestxray-dataset" in [configuration file](https://github.com/TebogoNakampe/TMIP-2019-nCoV-Recognition/blob/master/TMIP_Azure/config.yml)
	```bash
	cd ..
	bash tmip_data.sh
	find "$(cd ..; pwd)" -name "rsna-data" 
	find "$(cd ..; pwd)" -name "covid-chestxray-dataset" 
	
	```
* Preprocess Data
	```bash
	cd TMIP_Azure/
	bash tmip_preprocess.sh
	```
* Train ML Model
	```bash
	bash tmip_train.sh
	```
## oneAPI Environment Setup
* Request access to the  [oneAPI DevCloud ](https://software.intel.com/en-us/devcloud)
* Clone TMIP Repo
	```bash
	git clone https://github.com/TebogoNakampe/TMIP-2019-nCoV-Recognition.git
	cd TMIP-2019-nCoV-Recognition/TMIP_oneAPI/
	pip install -r requirements.txt
	```
* Get Data
	```bash
	cd ..
	bash tmip_data.sh
	```
* Preprocess Data
	```bash
	cd TMIP_Azure/
	bash tmip_preprocess.sh
	```
* Train ML Model
	```bash
	qsub -I -l walltime=24:00:00
	qsub -l nodes=4:gpu:ppn=2 -l walltime=24:00:00 -d . tmip.sh
	```
# Citation

If you find this useful, please cite our work as follows:

@article{tebogonakampe2020TMIP,
  author = {Tebogo Nakampe, Thabo Koee,
  title = {Treatise of Medical Image Processing v020},
  journal = {TMIPv020},
  year = {2020},
}

Please contact "info@4ir-abi.co.za" if you have any questions.


