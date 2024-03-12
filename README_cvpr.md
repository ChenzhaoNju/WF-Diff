# Wavelet-based Fourier Information Interaction with Frequency Diffusion Adjustment for Underwater Image Restoration  (CVPR'2024)

Chen Zhao, Weiling Cai, Chenyu Dong and Ziqi Zeng

[![paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2311.16845)

<hr />

> **Abstract:** *Underwater images are subject to intricate and diverse degradation, inevitably affecting the effectiveness of underwater visual tasks. However, most approaches primarily operate in the raw pixel space of images, which limits the exploration of the frequency characteristics of underwater images, leading to an inadequate utilization of deep models' representational capabilities in producing high-quality images. 
In this paper, we introduce a novel Underwater Image Enhancement (UIE) framework, named WF-Diff, designed to fully leverage the characteristics of frequency domain information and diffusion models.
WF-Diff consists of two detachable networks: Wavelet-based Fourier information interaction network (WFI2-net) and Frequency Residual Diffusion Adjustment Module (FRDAM). With our full exploration of the frequency domain information, WFI2-net aims to achieve preliminary enhancement of frequency information in the wavelet space. Our proposed FRDAM can further refine the high- and low-frequency information of the initial enhanced images, which can be viewed as a plug-and-play universal module to adjust the detail of the underwater images. With the above techniques, our algorithm can show SOTA performance on real-world underwater image datasets, and achieves competitive performance in visual quality.* 
<hr />

## Network Architecture


## Installation and Data Preparation



## Training

After preparing the training data in ```data/``` directory, use 
```
python train.py
```
to start the training of the model. 

```
python train.py
```

## Testing

After preparing the testing data in ```data/test/``` directory. 


```
python test.py 
```





## Results

临近毕业，遇到一大坨事，代码开源预计5月份左右~~~

Nearing graduation, encountering a bunch of things, the code open-source is expected around May~~~




## Citation
If you use our work, please consider citing:

  
    
    @article{zhao2023wavelet,
      title={Wavelet-based Fourier Information Interaction with Frequency Diffusion Adjustment for Underwater Image Restoration},
      author={Zhao, Chen and Cai, Weiling and Dong, Chenyu and Hu, Chengwei},
      journal={arXiv preprint arXiv:2311.16845},
      year={2023}
      
    


## Contact
Should you have any questions, please contact 2518628273@qq.com
 

