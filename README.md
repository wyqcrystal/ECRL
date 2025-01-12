# ECRL
The source code is for the paper: [Modeling Event-level Causal Representation for Video Classification](https://dl.acm.org/doi/abs/10.1145/3664647.3681547) accepted in ACM MM 2024 by Yuqing Wang, Lei Meng, Haokai Ma, Yuqing Wang, Haibei Huang, Xiangxu Meng.

## Overview
ECRL introduces an Event-level Causal Representation Learning method to enhance the model’s causal awareness of event information. The architecture of ECRL is illustrated in Fig. An event-level causal graph is constructed using the Frame-to-Video Causal Modeling (F2VCM) module, which explores the event correlations by finding the correlation between the foreground and background of the video frames; the Causality-aware Event-level Representation Inference (CERI) module eliminates background and data bias in video data by implementing causal intervention on the causal graph. This enables the model to find information directly related to event representation and the causal structure in video classification tasks.
![image](https://github.com/user-attachments/assets/f9440f9c-312c-4113-8052-8a7ebd571736)

## Dependencies

* Python 3.8.10
* PyTorch 1.12.0+cu102
* pytorch-lightning==1.6.5
* Torchvision==0.8.2
* Pandas==1.3.5

## BibTeX
If you find this work useful for your research, please kindly cite ECRL by:
> @inproceedings{wang2024modeling,
  title={Modeling Event-level Causal Representation for Video Classification},
  author={Wang, Yuqing and Meng, Lei and Ma, Haokai and Wang, Yuqing and Huang, Haibei and Meng, Xiangxu},
  booktitle={Proceedings of the 32nd ACM International Conference on Multimedia},
  pages={3936--3944},
  year={2024}
}



