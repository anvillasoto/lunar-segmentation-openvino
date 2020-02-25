# Image Segmentation on Artificial Lunar Landscape Dataset

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

Now that we explained the background and motivation behind this project, we now proceed with the challenges that arise from realizing the segmentation model.

### How difficult it is to collect moon terrain data

Deep learning and computer vision problems always rely on [vast amounts of data](https://www.ibm.com/topics/computer-vision) in order for the resulting models to be accurate. That is, in an acceptable degree of accuracy, resulting models capture the essence of what they represent or trying to solve. In our case, we want to have a feasible image segmentation model that can be applied to various use cases particularly in the creation of object detection or bounding box model to locate large boulders in real time fashion. In our case, only agencies like NASA are able to collect these sets of data like this dataset entitled [High-resolution Lunar Topography](https://pgda.gsfc.nasa.gov/products/54). Oftentimes, due to [limitations of the sensors and communication systems installed on these rovers](https://www.nasa.gov/mission_pages/LRO/news/LRO_twta.html) that capture significant amount terrain data and send lunar data periodically, data of interest is difficult to collect. 

Fortunately, there exists one labelled dataset of lunar landscape images that could be used for our purpose of employing machine learning approach to object detection or segmentation (see the dataset section of this document for more information about the dataset).

### How to make these models usable

Especially when we want to use the model locally on lunar rovers, it is obviously infeasible to run predictions to the cloud and get back to the rover. That is why real-time processing is necessary for applications such as these. In our case, our model must be accurate and fast enough to spit out predictions locally. Not to mention the issue of [energy efficiency](https://www.researchgate.net/publication/332463258_Low-Power_Computer_Vision_Status_Challenges_Opportunities) when it comes to building a pipelined approach in realizing these powerful systems. 

To overcome these challenges, we need to have a toolkit that solves these problems while also proactively support frameworks that produces state-of-the-art segmentation models that fit our purpose. 


## AI to the Edge and The Intel® OpenVINO™ Toolkit





## Dataset

