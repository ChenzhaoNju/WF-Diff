# Wavelet-based Fourier Information Interaction with Frequency Diffusion Adjustment for Underwater Image Restoration  (CVPR'2024)

Chen Zhao, Weiling Cai, Chenyu Dong and Ziqi Zeng

[![CVPR](https://img.shields.io/badge/CVPR-Paper-<COLOR>.svg)](https://openaccess.thecvf.com/content/CVPR2024/papers/Zhao_Wavelet-based_Fourier_Information_Interaction_with_Frequency_Diffusion_Adjustment_for_Underwater_CVPR_2024_paper.pdf)

<hr />

> **Abstract:** *Underwater images are subject to intricate and diverse degradation, inevitably affecting the effectiveness of underwater visual tasks. However, most approaches primarily operate in the raw pixel space of images, which limits the exploration of the frequency characteristics of underwater images, leading to an inadequate utilization of deep models' representational capabilities in producing high-quality images. 
In this paper, we introduce a novel Underwater Image Enhancement (UIE) framework, named WF-Diff, designed to fully leverage the characteristics of frequency domain information and diffusion models.
WF-Diff consists of two detachable networks: Wavelet-based Fourier information interaction network (WFI2-net) and Frequency Residual Diffusion Adjustment Module (FRDAM). With our full exploration of the frequency domain information, WFI2-net aims to achieve preliminary enhancement of frequency information in the wavelet space. Our proposed FRDAM can further refine the high- and low-frequency information of the initial enhanced images, which can be viewed as a plug-and-play universal module to adjust the detail of the underwater images. With the above techniques, our algorithm can show SOTA performance on real-world underwater image datasets, and achieves competitive performance in visual quality.* 
<hr />





Note: To facilitate easier replication and usage by the community, we have integrated the original code into the BasicIR framework. As a result, there are slight differences from the original paper. The updated visual results and pre-trained models can be found on [PA-Diff](<https://github.com/chenydong/PA-Diff>).

## Training

After preparing the training data, use 
```
CUDA_VISIBLE_DEVICES=1  python basicsr/train.py -opt options/train/train_Wfdiff.yml 
```


## Testing

After preparing the testing data, use 
```
CUDA_VISIBLE_DEVICES=1  python basicsr/test.py -opt options/test/test_wfdiff.yml
```


## Citation
If you use our work, please consider citing:

  
 
    @inproceedings{zhao2024wavelet,
    title={Wavelet-based fourier information interaction with frequency diffusion adjustment for underwater image restoration},
    author={Zhao, Chen and Cai, Weiling and Dong, Chenyu and Hu, Chengwei},
    booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
    pages={8281--8291}, year={2024}
    }



      
    


## Contact
Should you have any questions, please contact 2518628273@qq.com
 

