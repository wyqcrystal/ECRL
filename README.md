# ECRL
The source code is for the paper: Modeling Event-level Causal Representation for Video Classification accepted in ACM MM by Yuqing Wang, Lei Meng, Haokai Ma, Yuqing Wang, Haibei Huang, Xiangxu Meng.

## Overview
ECRL introduces an Event-level Causal Representation Learning method to enhance the model’s causal awareness of event information. The architecture of ECRL is illustrated in Fig. An event-level causal graph is constructed using the Frame-to-Video Causal Modeling (F2VCM) module, which explores the event correlations by finding the correlation between the foreground and background of the video frames; the Causality-aware Event-level Representation Inference (CERI) module eliminates background and data bias in video data by implementing causal intervention on the causal graph. This enables the model to find information directly related to event representation and the causal structure in video classification tasks.
![image](https://github.com/user-attachments/assets/8a96ac4e-0121-4cc7-85c4-181fdde8c63d)

## Dependencies

##

