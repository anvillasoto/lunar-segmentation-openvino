# Image Segmentation on Artificial Lunar Landscape Dataset

<img src="https://github.com/geochri/lunar-segmentation-openvino/blob/master/logo_lunarX.png" alt="logo" width="550"/>

## Abstract

An implementation of PyTorch's UNet Model for Image Segmentation on Artificial Lunar Landscape Dataset that also works on Intel® OpenVINO™


## Introduction

In space robotics particularly to the exploration of the moon, we rely on the power of lunar rovers, partially or fully autonomous vehicles that navigate on the surface of the moon. The obvious challenge on operating these vehicles is how we can safely navigate these into steep and rocky terrains. Like what Kobie Boykins, an engineer in JPL's spacecraft mechanical and engineering division said on his [2011 interview for Space.com](https://www.space.com/12482-moon-car-lunar-rover-apollo-15-legacy.html), we needed to then figure out how to drive over rocks and drive on steeper inclines. This could not have been any more true especially when these rovers are miles away from us, and we spend millions, if not billions for these space missions.

But this interview is about a decade ago. The very same year, we have had [incredible milestones towards deep learning and computer vision](forbes.com/sites/bernardmarr/2018/12/31/the-most-amazing-artificial-intelligence-milestones-so-far/#4968a43b7753) and these milestones kept pushing the limit as to what we can do across myriad of industries including space exploration. The year is 2020 as of this writing and so many deep learning applications are ready that we are now capable of improving these vehicles by aiding them with state-of-the-art computer vision models that would allow for detecting large rocks in a real-time fashion.

In this document, we will discuss the background and motivation of this study, the challenges that arise from creating a model for large rock detection which is part of smart navigation system we target the lunar rovers to have, how AI at the edge would help in solving these challenges and ultimately how we realized these lunar segmentation models by discussing the dataset, image segmentation techniques, model results and the end product using Intel® OpenVINO™ Toolkit.


## Background

The Trump administration is unequivocal in its pursuits of landing another man on the moon as well as building a sustainable infrastructure for long-term lunar and subsequently Martian explorations through its Artemis Program. The first bullet under Why Go to the Moon? section of [this](https://www.nasa.gov/what-is-artemis) article from NASA stipulates how this program will enable them to demonstrate new technologies, capabilities, and business approaches needed for future lunar exploration including Mars. 

In response to this, NASA itself [asks the general public and capable industries](https://www.nasa.gov/feature/nasa-to-industry-send-ideas-for-lunar-rovers) to send ideas on how to approach the development of robotic mobility systems and human-class lunar rovers to aid in conducting critical scientific research across wide areas of mostly unexplored lunar terrain. Within the same appeal article, Steve Clark, deputy associate administrator for exploration, Science Mission Directorate at NASA Headquarters in Washington said that “We are turning to industry to offer us exciting approaches to leverage existing systems here on Earth—including law enforcement, military, or recreational vehicles—that could be modified for use in space to enhance our mobility architecture.”

Another subset of Artemis Program is [Project VIPER](https://www.nasa.gov/feature/new-viper-lunar-rover-to-map-water-ice-on-the-moon) which aims to create a robust lunar rover that examines the presence of water on the moon in lieu with the aforesaid sustainability target. the rover will collect data on different kinds of soil environments, so obviously, this endeavor needs robust navigation systems especially when this exploration deals with uncharted terrains. 

In a research entitled [Robotics and Autonomous System by M.Novara P.Putz L.Maréchal S.Losito](https://www.sciencedirect.com/science/article/abs/pii/S092188909800058X), they stressed the importance of balance of autonomy and telepresence including the impact on technology development and operational costs, which are factors affecting robotics design. 


## Challenges
- Concept challenges:
The project we selected is by nature a challenging one. Since the moon terrain data is very difficult to be collected, the concept behind this research project is to use realistic artificial moon landscapes in order to match with the real ones. Aiming to adress this challenge, deep learning and computer vision techniques were employed, in order to help lunar robots on landscape detection. In the current project we used a modified unet to create an edge App with openvino package. The unet model is scripted with pytorch framework.

- Technical challeges:
One important technical challenge we had to adress, was the indirect compatibility between pytorch and OpenVINO package. The steps that we followed include the creation of the model using Pytorch, then the model is exported in ONNX format. Finally, a conversion of the ONNX model to OpenVINO, using Model Optimizer, is necessary. 

### How difficult it is to collect moon terrain data

As already mentioned, the lunar project, entails many difficulties. One of them was finding the appropriate dataset. As most planetary data, also moon terrain data are very sensitive and rare. This is attributed to two factors. The first one is related to the concept of a deep learning model. The need of big datasets is crucial in deep learning in order to achieve the optimal results. Therefore, in the case of moon terrain data, these big datasets are not applicable, as a robot is recording footsteps and this information is delivered to a satellite and afterwards to the robot's control center. The transfer speed is low and also time management is important as the robot should combine work and data transfer, thus a compromise between these two and a limitation in data, is inavoidable. The second restrictive factor is associated with the fact that our team is not a member of a space agency or an institution that would have a better access to an optimum sample of data. 

Deep learning and computer vision problems always rely on [vast amounts of data](https://www.ibm.com/topics/computer-vision) in order for the resulting models to be accurate. That is, in an acceptable degree of accuracy, resulting models capture the essence of what they represent or trying to solve. In our case, we want to have a feasible image segmentation model that can be applied to various use cases particularly in the creation of object detection or bounding box model to locate large boulders in real time fashion. In our case, only agencies like NASA are able to collect these sets of data like this dataset entitled [High-resolution Lunar Topography](https://pgda.gsfc.nasa.gov/products/54). Oftentimes, due to [limitations of the sensors and communication systems installed on these rovers](https://www.nasa.gov/mission_pages/LRO/news/LRO_twta.html) that capture significant amount terrain data and send lunar data periodically, data of interest is difficult to collect. 

Fortunately, there exists one labelled dataset of lunar landscape images that could be used for our purpose of employing machine learning approach to object detection or segmentation (see the dataset section of this document for more information about the dataset).



## Unet Topology
As mentioned above, in the current projectm, we used Unet Topology. Olaf Ronneberger et al. developed this model for Bio Medical Image Segmentation. The model's architecture is divided in two sections. The utility of the first part, also known as the contraction path (encoder), is to capture the context in the image. The encoder consists of convolutional and max poolong layers. The second part, also known as the decoder is responsible for the precise localization , with the use of transposed convolutions. It is a fully convolutional network, which consists of convolutional layers, without any dense layer, which enables it to accept images of any size. Upsampling operators that replace pooling operations, increase the resolution of the output. The prediction of the pixels in the border region of the image, is achived by extrapolating the missing context, by mirroring the input image. This tiling strategy enables the application of the network to large images, since otherwise the resolution would be limited by the GPU memory.

![unet topology-paper](https://github.com/geochri/lunar-segmentation-openvino/blob/master/unet_topology.png)


#### Semantic Segmentation
The objective of Semantic image Segmentation is to classify each pixel of an image, based on what it represents. This procedure is repeated and applied in every single pixel of an image, thus this task is also known as dense prediction. Contrary to other techniques, like image classification, classification with localization and object detection, semantic segmentation provides a high resolution image, of the same size as the input image, where each picture corresponds to a specific class. Therefore, in semantic segmentation the output is not labels and box parameters, but a pixel by pixel classification.

####  Semantic Segmentation-Applications

Some applications of Semantic Segmentation can be summarized as follows:
- Autonomous vehicles
- Bio Medical Image Diagnosis
- Geo Sensing
- Precision Agriculture

#### Why Unet
There are several advantages in using U-net for our project. First of all, considering the limited dataset sample we were dealing with, U-net provided the optimal results, as it has been tested as a segmentation tool in projects with small datasets, e.g. less than 50 training samples. Second, an also important feature of U-net is that it can be used with large images datasets, as it does not have any fully connected layers. Owing to this characteristic, features from images with different sizes, can be extracted. Summing the above benefits and considering the limitations we faced with our dataset, U-net was selected as the ideal segmentation tool for our lunar project.

### How to make these models usable

Especially when we want to use the model locally on lunar rovers, it is obviously infeasible to run predictions to the cloud and get back to the rover. That is why real-time processing is necessary for applications such as these. In our case, our model must be accurate and fast enough to spit out predictions locally. Not to mention the issue of [energy efficiency](https://www.researchgate.net/publication/332463258_Low-Power_Computer_Vision_Status_Challenges_Opportunities) when it comes to building a pipelined approach in realizing these powerful systems. 

To overcome these challenges, we need to have a toolkit that solves these problems while also proactively support frameworks that produces state-of-the-art segmentation models that fit our purpose. 

## AI at the Edge and The Intel® OpenVINO™ Toolkit

In response to the need for robust yet low-powered solutions for computer vision tasks that can be installed as an auxiliary system for autonomous systems, Intel released the OpenVINO toolkit to enable developers to do just that. Per the [documentation](https://software.intel.com/en-us/openvino-toolkit), OpenVINO enables deep learning inference right within edge, accelerates AI workloads, including computer vision with open support for many computer vision libraries such as OpenCV, OpenCL™, and other industry frameworks and libraries like PyTorch, Caffe, and TensorFlow.

This is beneficial for our purpose since our model which was written using PyTorch, needs to be converted into a system in such a way that it can run without cloud supervision, can be optimized using Intel hardware which are [currently being used by these autonomous vehicles for their computation requirements](https://newsroom.intel.com/news/intel-hosts-nasa-frontier-development-lab-demo-day-2018-research-presentations/#gs.xf1snl).


## Results

### Pytorch results
#### Untrained model
![Input Image](https://github.com/geochri/lunar-segmentation-openvino/blob/master/art_realistic_moon3.png)
![Untrained result](https://github.com/geochri/lunar-segmentation-openvino/blob/master/art_realistic_moon3_untrained_model.png)

#### Results after training
##### Example1 - input/ground truth/prediction
![Input Image1](https://github.com/geochri/lunar-segmentation-openvino/blob/master/art_realistic_moon.png)
![Ground truth1](https://github.com/geochri/lunar-segmentation-openvino/blob/master/art_realistic_moon_ground_truth.png)
![Prediction image1](https://github.com/geochri/lunar-segmentation-openvino/blob/master/art_realistic_moon_prediction.png)
##### Example2 - input/ground truth/prediction
![Input Image2](https://github.com/geochri/lunar-segmentation-openvino/blob/master/art_realistic_moon2.png)
![Ground truth2](https://github.com/geochri/lunar-segmentation-openvino/blob/master/art_realistic_moon_ground_truth2.png)
![Prediction image2](https://github.com/geochri/lunar-segmentation-openvino/blob/master/art_realistic_moon_prediction2.png)

##### Example3 - input/ground truth/prediction
![Input Image3](https://github.com/geochri/lunar-segmentation-openvino/blob/master/lunar_rock_segmentationV4_local_train.jpg)
![Ground truth3](https://github.com/geochri/lunar-segmentation-openvino/blob/master/lunar_rock_segmentationV4_local_mask.jpg)
![Prediction image3](https://github.com/geochri/lunar-segmentation-openvino/blob/master/lunar_rock_segmentationV4_local_pred.jpg)

##### Real moon prediction - input/prediction
![Input Image2](https://github.com/geochri/lunar-segmentation-openvino/blob/master/real_moon.png)
![Prediction image2](https://github.com/geochri/lunar-segmentation-openvino/blob/master/prediction_real_moon.png)



### Openvino Video presentation
![Segmentation Demo](https://github.com/geochri/lunar-segmentation-openvino/blob/master/demo.gif)


[Segmentation Demo-Video](https://github.com/anvillasoto/lunar-segmentation-openvino/blob/master/demo.mp4)


## Dataset

The dataset employed in the current project was created by the Ishigami Laboratory (Space Robotics Group) of Keio University, Japan (https://www.kaggle.com/romainpessia/artificial-lunar-rocky-landscape-dataset). In order the dataset to be realistic,  NASA's LRO LOLA Elevation Model (https://astrogeology.usgs.gov/search/details/Moon/LRO/LOLA/Lunar_LRO_LOLA_Global_LDEM_118m_Mar2014/cub) was used. 

The digital elevation model (DEM) is based on data from the Lunar Orbiter Laser Altimeter (LOLA; Smith et al., 2010), an instrument on the National Aeronautics and Space Agency (NASA) Lunar Reconnaissance Orbiter (LRO) spacecraft (Tooley et al., 2010). The created DEM represents more than 6.5 billion measurements gathered between July 2009 and July 2013, adjusted for consistency in the coordinate system described below, and then converted to lunar radii (Mazarico et al., 2012). Elevations were computed by subtracting the lunar reference radius of 1737.4 km from the surface radius measurements (LRO Project and LGCWG, 2008; Archinal et al., 2011). 
Then the dataset creator used Terragen 4 (https://planetside.co.uk/) to render realistic CG lunar environment, based on the above mentioned DEM to extract elevation. 

The dataset currently contains 9,766 realistic renders of rocky lunar landscapes, and their segmented equivalents (the 3 classes are the sky, smaller rocks, and larger rocks). A table of bounding boxes for all larger rocks and processed, cleaned-up ground truth images are also provided. We recommend that users check the "Understanding and Using the Dataset" kernel which contains additional information on the dataset.

## Further work
Implementing and testing SegNet, ENet, ICNet on openvino.

