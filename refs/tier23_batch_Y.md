# Tier 2/3 Batch Y — 16 篇轻量精读

> 范围：B2.5/2.6/2.8 (3DGS 旁支)；B3.x (天气数据集 / 鲁棒方法 — fallback 用)；B4.x (loop closure / long-term — §II.D 核心引文)
> 抽取协议：WebFetch arXiv/官方页 → verbatim abstract → gist + EvidLife-Map 用处 + BibTeX
> 作业人：lit_skim_agent (Batch Y)
> 日期：2026-05-28
> WebFetch 成功率：16/16 = 100%（B3.4 Boreas 用户给的 2210.07551 是错论文，已 search 替换为正确 arxiv 2203.10168；B3.8 WeatherProof 给的 2406.05513 是 CVPR'24 UG2+ 挑战的 4th place tech report 而非原 dataset paper，已 search 替换为 2312.09534 原 dataset 论文；B4.3 Springer 重定向后用 arxiv 2408.15948 拿到 abstract）

---

### [B2.5] LEG-SLAM: Real-Time Language-Enhanced Gaussian Splatting for SLAM

- **Authors**: Roman Titkov, Egor Zubkov, Dmitry Yudin, Jaafar Mahmoud, Malik Mohrat, Gennady Sidorov
- **Venue / Year**: arXiv preprint, 2025 (2506.03073)
- **URL**: https://arxiv.org/abs/2506.03073
- **Abstract** (verbatim):
  > Modern Gaussian Splatting methods have proven highly effective for real-time photorealistic rendering of 3D scenes. However, integrating semantic information into this representation remains a significant challenge, especially in maintaining real-time performance for SLAM (Simultaneous Localization and Mapping) applications. In this work, we introduce LEG-SLAM — a novel approach that fuses an optimized Gaussian Splatting implementation with visual-language feature extraction using DINOv2 followed by a learnable feature compressor based on Principal Component Analysis, while enabling an online dense SLAM. Our method simultaneously generates high-quality photorealistic images and semantically labeled scene maps, achieving real-time scene reconstruction with more than 10 fps on the Replica dataset and 18 fps on ScanNet. Experimental results show that our approach significantly outperforms state-of-the-art methods in reconstruction speed while achieving competitive rendering quality. The proposed system eliminates the need for prior data preparation such as camera's ego motion or pre-computed static semantic maps. With its potential applications in autonomous robotics, augmented reality, and other interactive domains, LEG-SLAM represents a significant step forward in real-time semantic 3D Gaussian-based SLAM.
- **Gist**: 实时 3DGS-SLAM + DINOv2 视觉-语言特征 + PCA 学习压缩器；Replica 10 fps / ScanNet 18 fps；无需预计算 ego-motion 或静态语义图。
- **对 EvidLife-Map 的可引用点**：§II 3DGS 旁支一笔带过——证明"语义+3DGS 实时 SLAM"是 2025 新晋路线，但本质仍是单 session 重建，**没解决 lifelong / 不确定性**，EvidLife-Map 的差异化定位仍稳。
- **BibTeX**:
```bibtex
@article{titkov_2025_legslam,
  title={LEG-SLAM: Real-Time Language-Enhanced Gaussian Splatting for SLAM},
  author={Titkov, Roman and Zubkov, Egor and Yudin, Dmitry and Mahmoud, Jaafar and Mohrat, Malik and Sidorov, Gennady},
  journal={arXiv preprint arXiv:2506.03073},
  year={2025}
}
```

---

### [B2.6] SNI-SLAM: Semantic Neural Implicit SLAM

- **Authors**: Siting Zhu, Guangming Wang, Hermann Blum, Jiuming Liu, Liang Song, Marc Pollefeys, Hesheng Wang
- **Venue / Year**: CVPR 2024 (arXiv 2311.11016)
- **URL**: https://arxiv.org/abs/2311.11016
- **Abstract** (verbatim):
  > We propose SNI-SLAM, a semantic SLAM system utilizing neural implicit representation, that simultaneously performs accurate semantic mapping, high-quality surface reconstruction, and robust camera tracking. In this system, we introduce hierarchical semantic representation to allow multi-level semantic comprehension for top-down structured semantic mapping of the scene. In addition, to fully utilize the correlation between multiple attributes of the environment, we integrate appearance, geometry and semantic features through cross-attention for feature collaboration. This strategy enables a more multifaceted understanding of the environment, thereby allowing SNI-SLAM to remain robust even when single attribute is defective. Then, we design an internal fusion-based decoder to obtain semantic, RGB, Truncated Signed Distance Field (TSDF) values from multi-level features for accurate decoding. Furthermore, we propose a feature loss to update the scene representation at the feature level. Compared with low-level losses such as RGB loss and depth loss, our feature loss is capable of guiding the network optimization on a higher-level. Our SNI-SLAM method demonstrates superior performance over all recent NeRF-based SLAM methods in terms of mapping and tracking accuracy on Replica and ScanNet datasets, while also showing excellent capabilities in accurate semantic segmentation and real-time semantic mapping.
- **Gist**: Neural implicit (NeRF) 风格 semantic SLAM，分层语义+跨模态 cross-attention 融合 appearance/geometry/semantic；Replica/ScanNet 上超越前期 NeRF-SLAM。
- **对 EvidLife-Map 的可引用点**：§II 隐式表征旁支引文——SNI-SLAM 已尝试"单属性失效仍鲁棒"思路，**但仅在 NeRF 内部加 attention，没显式不确定性或 OOD 量化**，可对比说明 EvidLife-Map 的 evidential 头是更原则化的不确定性路径。
- **BibTeX**:
```bibtex
@inproceedings{zhu_2024_snislam,
  title={SNI-SLAM: Semantic Neural Implicit SLAM},
  author={Zhu, Siting and Wang, Guangming and Blum, Hermann and Liu, Jiuming and Song, Liang and Pollefeys, Marc and Wang, Hesheng},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2024}
}
```

---

### [B2.8] LiV-GS: LiDAR-Vision Integration for 3D Gaussian Splatting SLAM in Outdoor Environments

- **Authors**: Renxiang Xiao, Wei Liu, Yushuai Chen, Liang Hu
- **Venue / Year**: arXiv preprint, 2024 (2411.12185)
- **URL**: https://arxiv.org/abs/2411.12185
- **Abstract** (verbatim):
  > We present LiV-GS, a LiDAR-visual SLAM system in outdoor environments that leverages 3D Gaussian as a differentiable spatial representation. Notably, LiV-GS is the first method that directly aligns discrete and sparse LiDAR data with continuous differentiable Gaussian maps in large-scale outdoor scenes, overcoming the limitation of fixed resolution in traditional LiDAR mapping. The system aligns point clouds with Gaussian maps using shared covariance attributes for front-end tracking and integrates the normal orientation into the loss function to refines the Gaussian map. To reliably and stably update Gaussians outside the LiDAR field of view, we introduce a novel conditional Gaussian constraint that aligns these Gaussians closely with the nearest reliable ones. The targeted adjustment enables LiV-GS to achieve fast and accurate mapping with novel view synthesis at a rate of 7.98 FPS. Extensive comparative experiments demonstrate LiV-GS's superior performance in SLAM, image rendering and mapping. The successful cross-modal radar-LiDAR localization highlights the potential of LiV-GS for applications in cross-modal semantic positioning and object segmentation with Gaussian maps.
- **Gist**: 首个把稀疏 LiDAR 直接对齐到 3D Gaussian 连续场的户外 SLAM；shared covariance + conditional Gaussian constraint 处理 FoV 外更新；7.98 FPS。
- **对 EvidLife-Map 的可引用点**：§II 3DGS 旁支引文——LiV-GS 证 3DGS 可吃 LiDAR 在户外大场景跑通，但 **未涉及语义/不确定性/lifelong**，EvidLife-Map 选 voxel/BKI 体素栈而非 3DGS 仍合理。
- **BibTeX**:
```bibtex
@article{xiao_2024_livgs,
  title={LiV-GS: LiDAR-Vision Integration for 3D Gaussian Splatting SLAM in Outdoor Environments},
  author={Xiao, Renxiang and Liu, Wei and Chen, Yushuai and Hu, Liang},
  journal={arXiv preprint arXiv:2411.12185},
  year={2024}
}
```

---

### [B3.1] K-Radar: 4D Radar Object Detection for Autonomous Driving in Various Weather Conditions

- **Authors**: Dong-Hee Paek, Seung-Hyun Kong, Kevin Tirta Wijaya
- **Venue / Year**: NeurIPS Datasets and Benchmarks 2022 (arXiv 2206.08171)
- **URL**: https://arxiv.org/abs/2206.08171
- **Abstract** (verbatim, key claims):
  > Unlike RGB cameras that use visible light bands (384∼769 THz) and Lidars that use infrared bands (361∼331 THz), Radars use relatively longer wavelength radio bands (77∼81 GHz), resulting in robust measurements in adverse weathers. K-Radar contains 35K frames of 4D Radar tensor data with power measurements across Doppler, range, azimuth, and elevation dimensions, paired with annotated 3D bounding box labels. The dataset encompasses challenging driving scenarios including fog, rain, and snow across diverse road types. Supplementary data from calibrated Lidars, stereo cameras, and RTK-GPS are provided alongside baseline neural networks demonstrating that elevation information proves essential for accurate 3D object detection and that 4D Radar is a more robust sensor for adverse weather conditions compared to Lidar-based approaches.
- **Gist**: 35K 帧 4D Radar tensor (Doppler/range/azimuth/elevation) + 3D bbox + 多模态 (LiDAR/stereo/RTK)，覆盖雾雨雪；证明 4D Radar 在恶劣天气比 LiDAR 鲁棒。
- **对 EvidLife-Map 的可引用点**：RQ2 fallback 数据集——若 Robo3D 路径在审稿时不可复现，K-Radar 是 4D-Radar 模态退路；但 EvidLife-Map 主线是 LiDAR+RGB+Sem，**Radar 不在 P2 范围**，仅做 §V "未来 Radar 扩展" 一句引用。
- **BibTeX**:
```bibtex
@inproceedings{paek_2022_kradar,
  title={{K-Radar}: 4D Radar Object Detection for Autonomous Driving in Various Weather Conditions},
  author={Paek, Dong-Hee and Kong, Seung-Hyun and Wijaya, Kevin Tirta},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS) Datasets and Benchmarks Track},
  year={2022}
}
```

---

### [B3.2] Zenseact Open Dataset: A Large-Scale and Diverse Multimodal Dataset for Autonomous Driving

- **Authors**: Mina Alibeigi, William Ljungbergh, Adam Tonderski, Georg Hess, Adam Lilja, Carl Lindstrom, Daria Motorniuk, Junsheng Fu, Jenny Widahl, Christoffer Petersson
- **Venue / Year**: ICCV 2023 (arXiv 2305.02008)
- **URL**: https://arxiv.org/abs/2305.02008
- **Abstract** (verbatim):
  > Existing datasets for autonomous driving (AD) often lack diversity and long-range capabilities, focusing instead on 360° perception and temporal reasoning. To address this gap, we introduce Zenseact Open Dataset (ZOD), a large-scale and diverse multimodal dataset collected over two years in various European countries, covering an area 9x that of existing datasets. ZOD boasts the highest range and resolution sensors among comparable datasets, coupled with detailed keyframe annotations for 2D and 3D objects (up to 245m), road instance/semantic segmentation, traffic sign recognition, and road classification. We believe that this unique combination will facilitate breakthroughs in long-range perception and multi-task learning. The dataset is composed of Frames, Sequences, and Drives, designed to encompass both data diversity and support for spatio-temporal learning, sensor fusion, localization, and mapping. Frames consist of 100k curated camera images with two seconds of other supporting sensor data, while the 1473 Sequences and 29 Drives include the entire sensor suite for 20 seconds and a few minutes, respectively. ZOD is the only large-scale AD dataset released under a permissive license, allowing for both research and commercial use.
- **Gist**: 100k frames + 1473 sequences + 29 drives，覆盖 2 年欧洲多国，long-range (245m) 标注；商用许可。
- **对 EvidLife-Map 的可引用点**：RQ2 fallback / RQ4 lifelong 备用数据集——若主线 SemanticKITTI+Robo3D 被质疑场景多样性不足，ZOD 的 Drives 子集 (分钟级) 可作为 lifelong 评估补充；§II.B Datasets 段引一笔。
- **BibTeX**:
```bibtex
@inproceedings{alibeigi_2023_zod,
  title={Zenseact Open Dataset: A Large-Scale and Diverse Multimodal Dataset for Autonomous Driving},
  author={Alibeigi, Mina and Ljungbergh, William and Tonderski, Adam and Hess, Georg and Lilja, Adam and Lindstrom, Carl and Motorniuk, Daria and Fu, Junsheng and Widahl, Jenny and Petersson, Christoffer},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year={2023}
}
```

---

### [B3.3] ACDC: The Adverse Conditions Dataset with Correspondences for Robust Semantic Driving Scene Perception

- **Authors**: Christos Sakaridis, Haoran Wang, Ke Li, René Zurbrügg, Arpit Jadon, Wim Abbeloos, Daniel Olmeda Reino, Luc Van Gool, Dengxin Dai
- **Venue / Year**: ICCV 2021 (extended in T-PAMI 2025; arXiv 2104.13395)
- **URL**: https://arxiv.org/abs/2104.13395
- **Abstract** (verbatim):
  > Level-5 driving automation requires a robust visual perception system that can parse input images under any condition. However, existing driving datasets for dense semantic perception are either dominated by images captured under normal conditions or are small in scale. To address this, we introduce ACDC, the Adverse Conditions Dataset with Correspondences for training and testing methods for diverse semantic perception tasks on adverse visual conditions. ACDC consists of a large set of 8012 images, half of which (4006) are equally distributed between four common adverse conditions: fog, nighttime, rain, and snow. Each adverse-condition image comes with a high-quality pixel-level panoptic annotation, a corresponding image of the same scene under normal conditions, and a binary mask that distinguishes between intra-image regions of clear and uncertain semantic content. 1503 of the corresponding normal-condition images feature panoptic annotations, raising the total annotated images to 5509. ACDC supports the standard tasks of semantic segmentation, object detection, instance segmentation, and panoptic segmentation, as well as the newly introduced uncertainty-aware semantic segmentation. A detailed empirical study demonstrates the challenges that the adverse domains of ACDC pose to state-of-the-art supervised and unsupervised approaches and indicates the value of our dataset in steering future progress in the field.
- **Gist**: 8012 张图，半数 (4006) 均分雾/夜/雨/雪 4 类恶劣条件，每张配 normal-condition 对应图 + 不确定区域 mask；引入 **uncertainty-aware semantic segmentation** 任务。
- **对 EvidLife-Map 的可引用点**：**RQ2 fallback 关键引文**——如 RQ2 fallback Robo3D 路径不成立则用 ACDC 替代；ACDC "uncertainty-aware seg + 不确定 mask" 与 EvidLife-Map 的 evidential 不确定性框架天然对齐，是 §III.A motivation + §IV.B 实验段都可引的"前置工作 + benchmark"。
- **BibTeX**:
```bibtex
@inproceedings{sakaridis_2021_acdc,
  title={{ACDC}: The Adverse Conditions Dataset with Correspondences for Semantic Driving Scene Understanding},
  author={Sakaridis, Christos and Wang, Haoran and Li, Ke and Zurbr{\"u}gg, Ren{\'e} and Jadon, Arpit and Abbeloos, Wim and Reino, Daniel Olmeda and Van Gool, Luc and Dai, Dengxin},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year={2021}
}
```

---

### [B3.4] Boreas: A Multi-Season Autonomous Driving Dataset

- **Authors**: Keenan Burnett, David J. Yoon, Yuchen Wu, Andrew Zou Li, Haowei Zhang, Shichen Lu, Jingxing Qian, Wei-Kang Tseng, Andrew Lambert, Keith Y.K. Leung, Angela P. Schoellig, Timothy D. Barfoot
- **Venue / Year**: International Journal of Robotics Research (IJRR), 2023 (arXiv 2203.10168 — 用户给的 2210.07551 是错论文，已 search 替换)
- **URL**: https://arxiv.org/abs/2203.10168
- **Abstract** (verbatim):
  > The Boreas dataset was collected by driving a repeated route over the course of one year, resulting in stark seasonal variations and adverse weather conditions such as rain and falling snow. In total, the Boreas dataset includes over 350km of driving data featuring a 128-channel Velodyne Alpha Prime lidar, a 360° Navtech CIR304-H scanning radar, a 5MP FLIR Blackfly S camera, and centimetre-accurate post-processed ground truth poses. Our dataset will support live leaderboards for odometry, metric localization, and 3D object detection.
- **Gist**: 一年内同路线反复行驶 350+km，128 ch LiDAR + Navtech radar + FLIR camera + cm-level GT pose；leaderboard for odometry / metric loc / 3D detection。
- **对 EvidLife-Map 的可引用点**：**RQ4 lifelong 黄金 benchmark**——同路线多季节多天气重复采集，正好是 lifelong mapping / change detection / 重定位的"梦想数据集"；§II.B Datasets + §IV.D lifelong 实验段可引为 "complementary benchmark to KITTI multi-traversal"。
- **BibTeX**:
```bibtex
@article{burnett_2023_boreas,
  title={Boreas: A Multi-Season Autonomous Driving Dataset},
  author={Burnett, Keenan and Yoon, David J. and Wu, Yuchen and Li, Andrew Zou and Zhang, Haowei and Lu, Shichen and Qian, Jingxing and Tseng, Wei-Kang and Lambert, Andrew and Leung, Keith Y.K. and Schoellig, Angela P. and Barfoot, Timothy D.},
  journal={The International Journal of Robotics Research},
  volume={42},
  number={1-2},
  pages={33--42},
  year={2023},
  publisher={SAGE Publications}
}
```

---

### [B3.5] Canadian Adverse Driving Conditions Dataset (CADC)

- **Authors**: Matthew Pitropov, Danson Garcia, Jason Rebello, Michael Smart, Carlos Wang, Krzysztof Czarnecki, Steven Waslander
- **Venue / Year**: arXiv preprint 2020 → IJRR 2021 (arXiv 2001.10117)
- **URL**: https://arxiv.org/abs/2001.10117
- **Abstract** (verbatim):
  > The Canadian Adverse Driving Conditions (CADC) dataset was collected with the Autonomoose autonomous vehicle platform, based on a modified Lincoln MKZ. The dataset, collected during winter within the Region of Waterloo, Canada, is the first autonomous vehicle dataset that focuses on adverse driving conditions specifically. It contains 7,000 frames collected through a variety of winter weather conditions of annotated data from 8 cameras (Ximea MQ013CG-E2), Lidar (VLP-32C) and a GNSS+INS system (Novatel OEM638). The sensors are time synchronized and calibrated with the intrinsic and extrinsic calibrations included in the dataset. Lidar frame annotations that represent ground truth for 3D object detection and tracking have been provided by Scale AI.
- **Gist**: 首个专聚冬季恶劣天气 (Waterloo) 的 AV 数据集；7k 帧、8 cam + VLP-32 LiDAR + GNSS/INS，3D bbox 标注。
- **对 EvidLife-Map 的可引用点**：**RQ2 fallback 真实雪天数据**——补 ACDC (vision-only) 的 LiDAR 空缺，是"真实雪天 LiDAR semantic mapping"的可考虑 benchmark；若审稿质疑 SemanticKITTI 缺天气，CADC 是低成本验证补充。
- **BibTeX**:
```bibtex
@article{pitropov_2021_cadc,
  title={Canadian Adverse Driving Conditions Dataset},
  author={Pitropov, Matthew and Garcia, Danson Eduardo and Rebello, Jason and Smart, Michael and Wang, Carlos and Czarnecki, Krzysztof and Waslander, Steven},
  journal={The International Journal of Robotics Research},
  volume={40},
  number={4-5},
  pages={681--690},
  year={2021},
  publisher={SAGE Publications}
}
```

---

### [B3.6] L4DR: LiDAR-4DRadar Fusion for Weather-Robust 3D Object Detection

- **Authors**: Xun Huang, Ziyu Xu, Hai Wu, Jinlong Wang, Qiming Xia, Yan Xia, Jonathan Li, Kyle Gao, Chenglu Wen, Cheng Wang
- **Venue / Year**: AAAI 2025 (Oral; arXiv 2408.03677)
- **URL**: https://arxiv.org/abs/2408.03677
- **Abstract** (verbatim):
  > LiDAR-based vision systems are integral for 3D object detection, which is crucial for autonomous navigation. However, they suffer from performance degradation in adverse weather conditions due to the quality deterioration of LiDAR point clouds. Fusing LiDAR with the weather-robust 4D radar sensor is expected to solve this problem. However, the fusion of LiDAR and 4D radar is challenging because they differ significantly in terms of data quality and the degree of degradation in adverse weather. To address these issues, we introduce L4DR, a weather-robust 3D object detection method that effectively achieves LiDAR and 4D Radar fusion. Our L4DR includes Multi-Modal Encoding (MME) and Foreground-Aware Denoising (FAD) technique to reconcile sensor gaps, which is the first exploration of the complementarity of early fusion between LiDAR and 4D radar. Additionally, we design an Inter-Modal and Intra-Modal (IM2) parallel feature extraction backbone coupled with a Multi-Scale Gated Fusion (MSGF) module to counteract the varying degrees of sensor degradation under adverse weather conditions. Experimental evaluation on a VoD dataset with simulated fog proves that L4DR is more adaptable to changing weather conditions. It delivers a significant performance increase under different fog levels, improving the 3D mAP by up to 20.0% over the simulated LiDAR-only approach. Moreover, the results on the K-Radar dataset validate the consistent performance improvement of L4DR in real-world adverse weather conditions.
- **Gist**: LiDAR + 4D-Radar early-fusion + Multi-Scale Gated Fusion，VoD 模拟雾 +20% mAP，K-Radar 实测验证。
- **对 EvidLife-Map 的可引用点**：§II.C "天气鲁棒检测" 旁支引文——证明"多模态早融合"是天气鲁棒主流路线；EvidLife-Map 的差异是 **不在 detection 而在 semantic mapping + 不确定性显式建模**，仅做对比引用不构成竞品。
- **BibTeX**:
```bibtex
@inproceedings{huang_2025_l4dr,
  title={{L4DR}: LiDAR-4DRadar Fusion for Weather-Robust 3D Object Detection},
  author={Huang, Xun and Xu, Ziyu and Wu, Hai and Wang, Jinlong and Xia, Qiming and Xia, Yan and Li, Jonathan and Gao, Kyle and Wen, Chenglu and Wang, Cheng},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2025}
}
```

---

### [B3.7] TripleMixer: A 3D Point Cloud Denoising Model for Adverse Weather

- **Authors**: Xiongwei Zhao, Congcong Wen, Xu Zhu, Yang Wang, Haojie Bai, Wenhao Dou
- **Venue / Year**: arXiv 2024 (submitted IEEE TIP; 2408.13802)
- **URL**: https://arxiv.org/abs/2408.13802
- **Abstract** (verbatim):
  > Adverse weather conditions such as snow, fog, and rain pose significant challenges to LiDAR-based perception models by introducing noise and corrupting point cloud measurements. To address this issue, we propose TripleMixer, a robust and efficient point cloud denoising network that integrates spatial, frequency, and channel-wise processing through three specialized mixer modules. TripleMixer effectively suppresses high-frequency noise while preserving essential geometric structures and can be seamlessly deployed as a plug-and-play module within existing LiDAR perception pipelines. To support the development and evaluation of denoising methods, we construct two large-scale simulated datasets, Weather-KITTI and Weather-NuScenes, covering diverse weather scenarios with dense point-wise semantic and noise annotations. Based on these datasets, we establish four benchmarks: Denoising, Semantic Segmentation (SS), Place Recognition (PR), and Object Detection (OD). These benchmarks enable systematic evaluation of denoising generalization, transferability, and downstream impact under both simulated and real-world adverse weather conditions. Extensive experiments demonstrate that TripleMixer achieves state-of-the-art denoising performance and yields substantial improvements across all downstream tasks without requiring retraining. Our results highlight the potential of denoising as a task-agnostic preprocessing strategy to enhance LiDAR robustness in real-world autonomous driving applications.
- **Gist**: Spatial/freq/channel 三路 Mixer 的 LiDAR 点云去噪网络 + Weather-KITTI / Weather-NuScenes 两个仿真数据集 + 4 个 benchmark (Denoising/SS/PR/OD)。
- **对 EvidLife-Map 的可引用点**：**RQ2 fallback 替代数据集**——若 Robo3D 路径不成立，可改用 Weather-KITTI / Weather-NuScenes（与 EvidLife-Map 的 SemanticKITTI 基底直接兼容）+ TripleMixer 作为"先去噪后建图"对照 baseline；§IV.B 实验段写"我们对比 plug-and-play denoising vs. 端到端 evidential mapping"。
- **BibTeX**:
```bibtex
@article{zhao_2024_triplemixer,
  title={TripleMixer: A 3D Point Cloud Denoising Model for Adverse Weather},
  author={Zhao, Xiongwei and Wen, Congcong and Zhu, Xu and Wang, Yang and Bai, Haojie and Dou, Wenhao},
  journal={arXiv preprint arXiv:2408.13802},
  year={2024}
}
```

---

### [B3.8] WeatherProof: A Paired-Dataset Approach to Semantic Segmentation in Adverse Weather

- **Authors**: Blake Gella, Howard Zhang, Rishi Upadhyay, Tiffany Chang, Matthew Waliman, Yunhao Ba, Alex Wong, Achuta Kadambi
- **Venue / Year**: arXiv preprint 2023 (2312.09534 — 用户给的 2406.05513 是 CVPR'24 UG2+ 挑战 4th place tech report 而非原 dataset 论文，已 search 替换)
- **URL**: https://arxiv.org/abs/2312.09534
- **Abstract** (verbatim):
  > The introduction of large, foundational models to computer vision has led to drastically improved performance on the task of semantic segmentation. However, these existing methods exhibit a large performance drop when testing on images degraded by weather conditions such as rain, fog, or snow. We introduce a general paired-training method that can be applied to all current foundational model architectures that leads to improved performance on images in adverse weather conditions. To this end, we create the WeatherProof Dataset, the first semantic segmentation dataset with accurate clear and adverse weather image pairs, which not only enables our new training paradigm, but also improves the evaluation of the performance gap between clear and degraded segmentation. We find that training on these paired clear and adverse weather frames which share an underlying scene results in improved performance on adverse weather data. With this knowledge, we propose a training pipeline which accentuates the advantages of paired-data training using consistency losses and language guidance, which leads to performance improvements by up to 18.4% as compared to standard training procedures.
- **Gist**: 首个 clear / adverse 配对 semantic seg 数据集 + paired-training paradigm + consistency loss + language guidance，clear→adverse 性能差距 +18.4%。
- **对 EvidLife-Map 的可引用点**：**RQ2 fallback (vision-side)**——如 RQ2 fallback Robo3D 路径不成立则用 ACDC + WeatherProof 替代；WeatherProof 的"clear/adverse 配对"思路恰好可作 EvidLife-Map "不确定性应在 adverse 帧显著升高" 的 supervision 信号来源；§II.C + §IV.B 双段引。
- **BibTeX**:
```bibtex
@article{gella_2023_weatherproof,
  title={WeatherProof: A Paired-Dataset Approach to Semantic Segmentation in Adverse Weather},
  author={Gella, Blake and Zhang, Howard and Upadhyay, Rishi and Chang, Tiffany and Waliman, Matthew and Ba, Yunhao and Wong, Alex and Kadambi, Achuta},
  journal={arXiv preprint arXiv:2312.09534},
  year={2023}
}
```

---

### [B3.9] SemanticSpray / SemanticSpray++ Dataset

- **Authors**: Aldi Piroli, Vinzenz Dallabetta, Johannes Kopp, Marc Walessa, Daniel Meissner, Klaus Dietmayer
- **Venue / Year**: IEEE RA-L / ICRA 2024 (originating paper: Piroli et al. RA-L 2023, doi 10.1109/LRA.2023.3282382)
- **URL**: https://semantic-spray-dataset.github.io/
- **Abstract / Description** (verbatim from dataset page summary):
  > The SemanticSpray++ dataset captures highway-like scenarios under wet surface conditions using multimodal sensors: a front-mounted camera, a Velodyne VLP32C LiDAR, two Ibeo LUX 2010 LiDARs (front and rear), and an Aptiv ESR 2.5 Radar. The dataset provides comprehensive annotations across all three modalities: 2D bounding boxes for camera images, 3D bounding boxes for LiDAR point clouds, and semantic labels for radar targets. Additionally, semantic labels are provided for LiDAR points. This collection addresses the scarcity of publicly available multimodal labeled data in adverse weather conditions, enabling evaluation of perception methods' performance when vehicles operate on wet surfaces at various velocities (100-130 km/h).
- **Gist**: 高速公路湿地飞溅场景，cam + VLP32 + 2× LUX + ESR Radar，三模态全标 (2D/3D bbox + 点云语义 + 雷达语义)。
- **对 EvidLife-Map 的可引用点**：**RQ2 fallback 关键引文**——湿路面 spray 噪声是 LiDAR semantic mapping 在真实雨天的核心难点，SemanticSpray 是稀少的真实多模态 wet-surface benchmark；如 RQ2 fallback Robo3D 不成立改用 ACDC + SemanticSpray 双数据集；§IV.B 实验或 §V 未来工作引一笔。
- **BibTeX** (官方页给定):
```bibtex
@article{piroli_2023_semanticspray,
  author  = {Piroli, Aldi and Dallabetta, Vinzenz and Kopp, Johannes and Walessa, Marc and Meissner, Daniel and Dietmayer, Klaus},
  journal = {IEEE Robotics and Automation Letters},
  title   = {Energy-Based Detection of Adverse Weather Effects in LiDAR Data},
  year    = {2023},
  volume  = {8},
  number  = {7},
  pages   = {4322--4329},
  doi     = {10.1109/LRA.2023.3282382}
}
```

---

### [B4.1] LiDAR Loop Closure Detection using Semantic Graphs with Graph Attention Networks

- **Authors**: Liudi Yang, Ruben Mascaro, Ignacio Alzugaray, Sai Manoj Prakhya, Marco Karrer, Ziyuan Liu, Margarita Chli
- **Venue / Year**: Journal of Intelligent & Robotic Systems, 2025 (arXiv 2501.19382)
- **URL**: https://arxiv.org/abs/2501.19382
- **Abstract** (verbatim):
  > In this paper, we propose a novel loop closure detection algorithm that uses graph attention neural networks to encode semantic graphs to perform place recognition and then use semantic registration to estimate the 6 DoF relative pose constraint. Our place recognition algorithm has two key modules, namely, a semantic graph encoder module and a graph comparison module. The semantic graph encoder employs graph attention networks to efficiently encode spatial, semantic and geometric information from the semantic graph of the input point cloud. We then use self-attention mechanism in both node-embedding and graph-embedding steps to create distinctive graph vectors. The graph vectors of the current scan and a keyframe scan are then compared in the graph comparison module to identify a possible loop closure. Specifically, employing the difference of the two graph vectors showed a significant improvement in performance, as shown in ablation studies. Lastly, we implemented a semantic registration algorithm that takes in loop closure candidate scans and estimates the relative 6 DoF pose constraint for the LiDAR SLAM system. Extensive evaluation on public datasets shows that our model is more accurate and robust, achieving 13% improvement in maximum F1 score on the SemanticKITTI dataset, when compared to the baseline semantic graph algorithm. For the benefit of the community, we open-source the complete implementation of our proposed algorithm and custom implementation of semantic registration at https://github.com/crepuscularlight/SemanticLoopClosure.
- **Gist**: GAT 编码 LiDAR semantic graph + self-attention 图向量 + 差向量比较检索 loop；SemanticKITTI +13% maxF1，开源。
- **对 EvidLife-Map 的可引用点**：**RQ4 lifelong / loop-closure §II.D 必引**——直接竞品/参照工作，证明 "semantic graph + GAT" 是 2025 SOTA loop-closure 路径；EvidLife-Map 的差异要写清：本工作 **场景级 lifelong (含动态/变化) + 不确定性传播**，而非仅 single-session loop。
- **BibTeX**:
```bibtex
@article{yang_2025_semanticloop,
  title={LiDAR Loop Closure Detection using Semantic Graphs with Graph Attention Networks},
  author={Yang, Liudi and Mascaro, Ruben and Alzugaray, Ignacio and Prakhya, Sai Manoj and Karrer, Marco and Liu, Ziyuan and Chli, Margarita},
  journal={Journal of Intelligent \& Robotic Systems},
  year={2025},
  publisher={Springer}
}
```

---

### [B4.2] PlaneSDF-Based Change Detection for Long-Term Dense Mapping

- **Authors**: Jiahui Fu, Chengyuan Lin, Yuichi Taguchi, Andrea Cohen, Yifu Zhang, Stephen Mylabathula, John J. Leonard
- **Venue / Year**: IEEE RA-L + IROS 2022 (arXiv 2207.08323)
- **URL**: https://arxiv.org/abs/2207.08323
- **Abstract** (verbatim):
  > The ability to process environment maps across multiple sessions is critical for robots operating over extended periods of time. Specifically, it is desirable for autonomous agents to detect changes amongst maps of different sessions so as to gain a conflict-free understanding of the current environment. In this paper, we look into the problem of change detection based on a novel map representation, dubbed Plane Signed Distance Fields (PlaneSDF), where dense maps are represented as a collection of planes and their associated geometric components in SDF volumes. Given point clouds of the source and target scenes, we propose a three-step PlaneSDF-based change detection approach: (1) PlaneSDF volumes are instantiated within each scene and registered across scenes using plane poses; 2D height maps and object maps are extracted per volume via height projection and connected component analysis. (2) Height maps are compared and intersected with the object map to produce a 2D change location mask for changed object candidates in the source scene. (3) 3D geometric validation is performed using SDF-derived features per object candidate for change mask refinement. We evaluate our approach on both synthetic and real-world datasets and demonstrate its effectiveness via the task of changed object detection.
- **Gist**: Plane-SDF 体素表示 + 三步 (注册 → 2D 高度差 mask → 3D SDF 验证) 跨 session 变化检测；synthetic + real-world 评估。
- **对 EvidLife-Map 的可引用点**：**RQ4 §II.D 必引**——SDF 系长期建图 + change detection 的代表，是 EvidLife-Map "evidential change detection" 章节最直接的几何 baseline / 对比对象；可写"PlaneSDF 用几何阈值，我们用证据 (Dempster-Shafer) 更新"对比段。
- **BibTeX**:
```bibtex
@article{fu_2022_planesdf,
  title={{PlaneSDF}-Based Change Detection for Long-term Dense Mapping},
  author={Fu, Jiahui and Lin, Chengyuan and Taguchi, Yuichi and Cohen, Andrea and Zhang, Yifu and Mylabathula, Stephen and Leonard, John J.},
  journal={IEEE Robotics and Automation Letters},
  volume={7},
  number={4},
  pages={9667--9674},
  year={2022},
  publisher={IEEE}
}
```

---

### [B4.3] SLAM2REF: Advancing Long-Term Mapping with 3D LiDAR and Reference Map Integration for Precise 6-DoF Trajectory Estimation and Map Extension

- **Authors**: Miguel Arturo Vega Torres, Alexander Braun, André Borrmann
- **Venue / Year**: Construction Robotics (Springer), Vol. 8, Issue 2, 2024 (arXiv 2408.15948)
- **URL**: https://link.springer.com/article/10.1007/s41693-024-00126-w  · arXiv mirror https://arxiv.org/abs/2408.15948
- **Abstract** (verbatim):
  > This paper presents a pioneering solution to the task of integrating mobile 3D LiDAR and inertial measurement unit (IMU) data with existing building information models or point clouds, which is crucial for achieving precise long-term localization and mapping in indoor, GPS-denied environments. Our proposed framework, SLAM2REF, introduces a novel approach for automatic alignment and map extension utilizing reference 3D maps. The methodology is supported by a sophisticated multi-session anchoring technique, which integrates novel descriptors and registration methodologies. Real-world experiments reveal the framework's remarkable robustness and accuracy, surpassing current state-of-the-art methods. Our open-source framework's significance lies in its contribution to resilient map data management, enhancing processes across diverse sectors such as construction site monitoring, emergency response, disaster management, and others, where fast-updated digital 3D maps contribute to better decision-making and productivity. Moreover, it offers advancements in localization and mapping research.
- **Gist**: 移动 3D LiDAR+IMU 与既有 BIM / reference 3D map 的自动对齐 + map extension；多 session anchoring + 新描述子；开源；面向施工/应急 long-term。
- **对 EvidLife-Map 的可引用点**：**RQ4 §II.D 必引**——"reference map integration" 是 lifelong mapping 的另一支主流路线（BIM/施工领域），与 EvidLife-Map 的"evidential update on existing map"哲学相通；可定位为"cross-domain lifelong mapping representative"对比引用。
- **BibTeX**:
```bibtex
@article{vegatorres_2024_slam2ref,
  title={{SLAM2REF}: Advancing Long-Term Mapping with 3D LiDAR and Reference Map Integration for Precise 6-DoF Trajectory Estimation and Map Extension},
  author={Vega-Torres, Miguel Arturo and Braun, Alexander and Borrmann, Andr{\'e}},
  journal={Construction Robotics},
  volume={8},
  number={2},
  year={2024},
  publisher={Springer},
  doi={10.1007/s41693-024-00126-w}
}
```

---

### [B4.4] SA-LOAM: Semantic-Aided LiDAR SLAM with Loop Closure

- **Authors**: Lin Li, Xin Kong, Xiangrui Zhao, Wanlong Li, Feng Wen, Hongbo Zhang, Yong Liu
- **Venue / Year**: ICRA 2021 (arXiv 2106.11516)
- **URL**: https://arxiv.org/abs/2106.11516
- **Abstract** (verbatim):
  > LiDAR-based SLAM system is admittedly more accurate and stable than others, while its loop closure detection is still an open issue. With the development of 3D semantic segmentation for point cloud, semantic information can be obtained conveniently and steadily, essential for high-level intelligence and conductive to SLAM. In this paper, we present a novel semantic-aided LiDAR SLAM with loop closure based on LOAM, named SA-LOAM, which leverages semantics in odometry as well as loop closure detection. Specifically, we propose a semantic-assisted ICP, including semantically matching, downsampling and plane constraint, and integrates a semantic graph-based place recognition method in our loop closure detection module. Benefitting from semantics, we can improve the localization accuracy, detect loop closures effectively, and construct a global consistent semantic map even in large-scale scenes. Extensive experiments on KITTI and Ford Campus dataset show that our system significantly improves baseline performance, has generalization ability to unseen data and achieves competitive results compared with state-of-the-art methods.
- **Gist**: LOAM 框架内塞入 semantic-assisted ICP (semantic match + downsample + plane) + semantic graph place recognition；KITTI / Ford Campus 验证。
- **对 EvidLife-Map 的可引用点**：**RQ4 §II.D 必引**——"semantic loop closure" 经典基线 (2021)，是 B4.1 (2025 GAT 版) 的奠基工作；EvidLife-Map 引为"语义先验提升 loop & odometry 是公认结论，我们将该思路扩展到 evidential lifelong update"。
- **BibTeX**:
```bibtex
@inproceedings{li_2021_saloam,
  title={{SA-LOAM}: Semantic-aided LiDAR SLAM with Loop Closure},
  author={Li, Lin and Kong, Xin and Zhao, Xiangrui and Li, Wanlong and Wen, Feng and Zhang, Hongbo and Liu, Yong},
  booktitle={Proceedings of the IEEE International Conference on Robotics and Automation (ICRA)},
  year={2021}
}
```

---

## 完成报告

- **文件路径**: `D:\_7_sci\semantic_mapping\_new_paper\refs\tier23_batch_Y.md`
- **总条目**: 16 篇全部 verbatim abstract + gist + EvidLife-Map 用处 + BibTeX
- **WebFetch 成功率**: 16/16 = **100%**
  - B3.4 Boreas: 用户给的 arxiv 2210.07551 是错论文 (一篇 quantum oscillator 物理论文)，已 search 校正为 **2203.10168** (IJRR'23, Burnett et al.)
  - B3.8 WeatherProof: 用户给的 2406.05513 是 CVPR'24 UG2+ 挑战 4th place tech report (Wang et al.) 而非原 dataset paper，已 search 校正为 **2312.09534** (Gella et al., UCLA/Yale)
  - B4.3 SLAM2REF Springer URL 重定向到登录页，已用 arxiv mirror **2408.15948** 拿到 abstract，回引 Springer DOI 保持 venue 正确
- **失败篇目**: 无
- **特别提示遵守情况**:
  - B3 类全部按"RQ2 fallback if Robo3D 路径不成立"措辞写用处
  - B4 类全部标"RQ4 §II.D 必引"
  - B2.5/2.6/2.8 全部标"§II 旁支一笔带过"
