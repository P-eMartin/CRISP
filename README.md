# CRISP
## ComputeR vIsion for Sport Performance 

The goal of the project is to increase sportsmen performances by using Digital technologies. Sportsmen actions are recorded in an ecological situation of their exercises and games.
The first targeted sport is Table Tennis. The goal of automtic analysis is to index video recordings by recognising table tennis strokes in them. An ecological corpus is being recorded by students of Sport Faculty of University of Bordeaux, teachers and table tennis players. The corpus is annotated by experts via crowd-sourced interface. The methodology of action recognition is based on specifically designed Deep Learning architecture and motion analysis. A Fine-grain characterization of actions is then foreseen to optimize performances of sportsmen.
The goal of this repository is to allow research team, PhD students, to be able to reproduce our work, method in the aim to compare our method with theirs or enriched ours. If you use our code, please cite our work:

``
Pierre-Etienne  Martin,   Jenny  Benois-Pineau,   Renaud Péteri, and Julien Morlier, “Sport action recognition with siamese spatio-temporal cnns:  Application to table tennis,” in CBMI 2018. 2018, pp. 1–6, IEEE.
``

[bib source](MartinBPM18.bib)

# Dataset
## TTStroke-21

The dataset has been annotated using 20 stroke classes and a rejection class.
The dataset contains private data and is available through MediaEval workshop where we organize the Sport task based on TTStroke-21. To have access to the data, particular conditions need to be accepted. We are working on sharing this dataset while respecting the General Data Protection Regulation (EU GDPR).

# Method
## Twin Spatio-Temporal Concvolutional Neural Network (TSTCNN)

Our network take as input the optical flow computed from the rgb images and the rgb data. The size of the input data is set to (W x H x T) = (120 x 120 x 100).


### coming soon...

We are waiting for the authorization of the University and all the tutelles to share the code. Due to the COVID19 situation, this process has been delayed. Once we have the green light, the code will be available.
