
Weekly Classified Neural Radiance Fields - video ![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)
=================================================================================================================================================================
## Filter by classes: 
 [all](../weekly_nerf.md) | [dynamic](./dynamic.md) | [editing](./editing.md) | [fast](./fast.md) | [generalization](./generalization.md) | [human](./human.md) | [video](./video.md) | [lighting](./lighting.md) | [reconstruction](./reconstruction.md) | [texture](./texture.md) | [semantic](./semantic.md) | [pose-slam](./pose-slam.md) | [others](./others.md) 
## Dec27 - Jan3, 2023
  - [Boosting UAV Tracking with Voxel-Based Trajectory-Aware Pre-Training, RAL2022](https://ieeexplore.ieee.org/abstract/document/10015867) | [code]
    > Siamese network-based object tracking has remarkably promoted the automatic capability for highly-maneuvered unmanned aerial vehicles (UAVs). However, the leading-edge tracking framework often depends on template matching, making it trapped when facing multiple views of object in consecutive frames. Moreover, the general image-level pretrained backbone can overfit to holistic representations, causing the misalignment to learn object-level properties in UAV tracking. To tackle these issues, this work presents TRTrack , a comprehensive framework to fully exploit the stereoscopic representation for UAV tracking. Specifically, a novel pre-training paradigm method is proposed. Through trajectory-aware reconstruction training (TRT), the capability of the backbone to extract stereoscopic structure feature is strengthened without any parameter increment. Accordingly, an innovative hierarchical self-attention Transformer is proposed to capture the local detail information and global structure knowledge. For optimizing the correlation map, we proposed a novel spatial correlation refinement (SCR) module, which promotes the capability of modeling the long-range spatial dependencies. Comprehensive experiments on three UAV challenging benchmarks demonstrate that the proposed TRTrack achieves superior UAV tracking performance in both precision and efficiency. Quantitative tests in real-world settings fully prove the effectiveness of our work.
## Dec25 - Dec31, 2022
## Dec18 - Dec24, 2022
## Dec11 - Dec17, 2022
## Dec4 - Dec10, 2022
## Nov27 - Dec3, 2022
  - [QuadStream: A Quad-Based Scene Streaming Architecture for Novel Viewpoint Reconstruction, ToG2022](https://dl.acm.org/doi/abs/10.1145/3550454.3555524) | [code]
    > Streaming rendered 3D content over a network to a thin client device, such as a phone or a VR/AR headset, brings high-fidelity graphics to platforms where it would not normally possible due to thermal, power, or cost constraints. Streamed 3D content must be transmitted with a representation that is both robust to latency and potential network dropouts. Transmitting a video stream and reprojecting to correct for changing viewpoints fails in the presence of disocclusion events; streaming scene geometry and performing high-quality rendering on the client is not possible on limited-power mobile GPUs. To balance the competing goals of disocclusion robustness and minimal client workload, we introduce QuadStream, a new streaming content representation that reduces motion-to-photon latency by allowing clients to efficiently render novel views without artifacts caused by disocclusion events. Motivated by traditional macroblock approaches to video codec design, we decompose the scene seen from positions in a view cell into a series of quad proxies, or view-aligned quads from multiple views. By operating on a rasterized G-Buffer, our approach is independent of the representation used for the scene itself; the resulting QuadStream is an approximate geometric representation of the scene that can be reconstructed by a thin client to render both the current view and nearby adjacent views. Our technical contributions are an efficient parallel quad generation, merging, and packing strategy for proxy views covering potential client movement in a scene; a packing and encoding strategy that allows masked quads with depth information to be transmitted as a frame-coherent stream; and an efficient rendering approach for rendering our QuadStream representation into entirely novel views on thin clients. We show that our approach achieves superior quality compared both to video data streaming methods, and to geometry-based streaming.
## Nov20 - Nov26, 2022
## Nov13 - Nov19, 2022
## Nov6 - Nov12, 2022
## Oct30 - Nov5, 2022
## Oct23 - Oct29, 2022
## Oct16 - Oct22, 2022
## Oct9 - Oct15, 2022
## Oct2 - Oct8, 2022
## Sep25 - Oct1, 2022
  - [MonoNeuralFusion: Online Monocular Neural 3D Reconstruction with Geometric Priors](https://arxiv.org/abs/2209.15153) | [code]
    > High-fidelity 3D scene reconstruction from monocular videos continues to be challenging, especially for complete and fine-grained geometry reconstruction. The previous 3D reconstruction approaches with neural implicit representations have shown a promising ability for complete scene reconstruction, while their results are often over-smooth and lack enough geometric details. This paper introduces a novel neural implicit scene representation with volume rendering for high-fidelity online 3D scene reconstruction from monocular videos. For fine-grained reconstruction, our key insight is to incorporate geometric priors into both the neural implicit scene representation and neural volume rendering, thus leading to an effective geometry learning mechanism based on volume rendering optimization. Benefiting from this, we present MonoNeuralFusion to perform the online neural 3D reconstruction from monocular videos, by which the 3D scene geometry is efficiently generated and optimized during the on-the-fly 3D monocular scanning. The extensive comparisons with state-of-the-art approaches show that our MonoNeuralFusion consistently generates much better complete and fine-grained reconstruction results, both quantitatively and qualitatively.
## Sep18 - Sep24, 2022
## Sep11 - Sep17, 2022
## Sep4 - Sep10, 2022
## Aug28 - Sep3, 2022
## Aug21 - Aug27, 2022
## Aug14 - Aug20, 2022
  - [Temporal View Synthesis of Dynamic Scenes through 3D Object Motion Estimation with Multi-Plane Images, ISMAR2022](https://arxiv.org/abs/2208.09463) | [***``[code]``***](https://github.com/NagabhushanSN95/DeCOMPnet)
    > The challenge of graphically rendering high frame-rate videos on low compute devices can be addressed through periodic prediction of future frames to enhance the user experience in virtual reality applications. This is studied through the problem of temporal view synthesis (TVS), where the goal is to predict the next frames of a video given the previous frames and the head poses of the previous and the next frames. In this work, we consider the TVS of dynamic scenes in which both the user and objects are moving. We design a framework that decouples the motion into user and object motion to effectively use the available user motion while predicting the next frames. We predict the motion of objects by isolating and estimating the 3D object motion in the past frames and then extrapolating it. We employ multi-plane images (MPI) as a 3D representation of the scenes and model the object motion as the 3D displacement between the corresponding points in the MPI representation. In order to handle the sparsity in MPIs while estimating the motion, we incorporate partial convolutions and masked correlation layers to estimate corresponding points. The predicted object motion is then integrated with the given user or camera motion to generate the next frame. Using a disocclusion infilling module, we synthesize the regions uncovered due to the camera and object motion. We develop a new synthetic dataset for TVS of dynamic scenes consisting of 800 videos at full HD resolution. We show through experiments on our dataset and the MPI Sintel dataset that our model outperforms all the competing methods in the literature.
  - [Neural Capture of Animatable 3D Human from Monocular Video, ECCV2022](https://arxiv.org/abs/2208.08728) | [code]
    > We present a novel paradigm of building an animatable 3D human representation from a monocular video input, such that it can be rendered in any unseen poses and views. Our method is based on a dynamic Neural Radiance Field (NeRF) rigged by a mesh-based parametric 3D human model serving as a geometry proxy. Previous methods usually rely on multi-view videos or accurate 3D geometry information as additional inputs; besides, most methods suffer from degraded quality when generalized to unseen poses. We identify that the key to generalization is a good input embedding for querying dynamic NeRF: A good input embedding should define an injective mapping in the full volumetric space, guided by surface mesh deformation under pose variation. Based on this observation, we propose to embed the input query with its relationship to local surface regions spanned by a set of geodesic nearest neighbors on mesh vertices. By including both position and relative distance information, our embedding defines a distance-preserved deformation mapping and generalizes well to unseen poses. To reduce the dependency on additional inputs, we first initialize per-frame 3D meshes using off-the-shelf tools and then propose a pipeline to jointly optimize NeRF and refine the initial mesh. Extensive experiments show our method can synthesize plausible human rendering results under unseen poses and views.
  - [Casual Indoor HDR Radiance Capture from Omnidirectional Images](https://arxiv.org/abs/2208.07903) | [code]
    > We present PanoHDR-NeRF, a novel pipeline to casually capture a plausible full HDR radiance field of a large indoor scene without elaborate setups or complex capture protocols. First, a user captures a low dynamic range (LDR) omnidirectional video of the scene by freely waving an off-the-shelf camera around the scene. Then, an LDR2HDR network uplifts the captured LDR frames to HDR, subsequently used to train a tailored NeRF++ model. The resulting PanoHDR-NeRF pipeline can estimate full HDR panoramas from any location of the scene. Through experiments on a novel test dataset of a variety of real scenes with the ground truth HDR radiance captured at locations not seen during training, we show that PanoHDR-NeRF predicts plausible radiance from any scene point. We also show that the HDR images produced by PanoHDR-NeRF can synthesize correct lighting effects, enabling the augmentation of indoor scenes with synthetic objects that are lit correctly.
## Aug7 - Aug13, 2022
  - [PS-NeRV: Patch-wise Stylized Neural Representations for Videos](https://arxiv.org/abs/2208.03742) | [code]
    > We study how to represent a video with implicit neural representations (INRs). Classical INRs methods generally utilize MLPs to map input coordinates to output pixels. While some recent works have tried to directly reconstruct the whole image with CNNs. However, we argue that both the above pixel-wise and image-wise strategies are not favorable to video data. Instead, we propose a patch-wise solution, PS-NeRV, which represents videos as a function of patches and the corresponding patch coordinate. It naturally inherits the advantages of image-wise methods, and achieves excellent reconstruction performance with fast decoding speed. The whole method includes conventional modules, like positional embedding, MLPs and CNNs, while also introduces AdaIN to enhance intermediate features. These simple yet essential changes could help the network easily fit high-frequency details. Extensive experiments have demonstrated its effectiveness in several video-related tasks, such as video compression and video inpainting.
## Jul31 - Aug6, 2022
## Jul24 - Jul30, 2022
## Previous weeks
  - [﻿Plenoxels: Radiance Fields without Neural Networks, CVPR2022(oral)](https://arxiv.org/abs/2112.05131) | [***``[code]``***](https://alexyu.net/plenoxels)
    > We introduce Plenoxels (plenoptic voxels), a system for photorealistic view synthesis. Plenoxels represent a scene as a sparse 3D grid with spherical harmonics. This representation can be optimized from calibrated images via gradient methods and regularization without any neural components. On standard, benchmark tasks, Plenoxels are optimized two orders of magnitude faster than Neural Radiance Fields with no loss in visual quality.
  - [Neural Scene Flow Fields for Space-Time View Synthesis of Dynamic Scenes, CVPR2021](http://www.cs.cornell.edu/~zl548/NSFF/) | [***``[code]``***](https://github.com/zhengqili/Neural-Scene-Flow-Fields)
    > We present a method to perform novel view and time synthesis of dynamic scenes, requiring only a monocular video with known camera poses as input. To do this, we introduce Neural Scene Flow Fields, a new representation that models the dynamic scene as a time-variant continuous function of appearance, geometry, and 3D scene motion. Our representation is optimized through a neural network to fit the observed input views. We show that our representation can be used for complex dynamic scenes, including thin structures, view-dependent effects, and natural degrees of motion. We conduct a number of experiments that demonstrate our approach significantly outperforms recent monocular view synthesis methods, and show qualitative results of space-time view synthesis on a variety of real-world videos.
  - [Neural 3D Video Synthesis from Multi-view Video, CVPR2022(oral)](https://neural-3d-video.github.io/) | [code]
    > We propose a novel approach for 3D video synthesis that is able to represent multi-view video recordings of a dynamic real-world scene in a compact, yet expressive representation that enables high-quality view synthesis and motion interpolation. Our approach takes the high quality and compactness of static neural radiance fields in a new direction: to a model-free, dynamic setting. At the core of our approach is a novel time-conditioned neural radiance fields that represents scene dynamics using a set of compact latent codes. To exploit the fact that changes between adjacent frames of a video are typically small and locally consistent, we propose two novel strategies for efficient training of our neural network: 1) An efficient hierarchical training scheme, and 2) an importance sampling strategy that selects the next rays for training based on the temporal variation of the input videos. In combination, these two strategies significantly boost the training speed, lead to fast convergence of the training process, and enable high quality results. Our learned representation is highly compact and able to represent a 10 second 30 FPS multi-view video recording by 18 cameras with a model size of just 28MB. We demonstrate that our method can render high-fidelity wide-angle novel views at over 1K resolution, even for highly complex and dynamic scenes. We perform an extensive qualitative and quantitative evaluation that shows that our approach outperforms the current state of the art. Project website: https://neural-3d-video.github.io.
  - [Dynamic View Synthesis from Dynamic Monocular Video, ICCV2021](https://free-view-video.github.io/) | [***``[code]``***](https://github.com/gaochen315/DynamicNeRF)
    > We present an algorithm for generating novel views at arbitrary viewpoints and any input time step given a monocular video of a dynamic scene. Our work builds upon recent advances in neural implicit representation and uses continuous and differentiable functions for modeling the time-varying structure and the appearance of the scene. We jointly train a time-invariant static NeRF and a time-varying dynamic NeRF, and learn how to blend the results in an unsupervised manner. However, learning this implicit function from a single video is highly ill-posed (with infinitely many solutions that match the input video). To resolve the ambiguity, we introduce regularization losses to encourage a more physically plausible solution. We show extensive quantitative and qualitative results of dynamic view synthesis from casually captured videos.
  - [Editable Free-Viewpoint Video using a Layered Neural Representation, SIGGRAPH2021](https://jiakai-zhang.github.io/st-nerf/) | [***``[code]``***](https://jiakai-zhang.github.io/st-nerf/#code)
    > Generating free-viewpoint videos is critical for immersive VR/AR experience but recent neural advances still lack the editing ability to manipulate the visual perception for large dynamic scenes. To fill this gap, in this paper we propose the first approach for editable photo-realistic free-viewpoint video generation for large-scale dynamic scenes using only sparse 16 cameras. The core of our approach is a new layered neural representation, where each dynamic entity including the environment itself is formulated into a space-time coherent neural layered radiance representation called ST-NeRF. Such layered representation supports fully perception and realistic manipulation of the dynamic scene whilst still supporting a free viewing experience in a wide range. In our ST-NeRF, the dynamic entity/layer is represented as continuous functions, which achieves the disentanglement of location, deformation as well as the appearance of the dynamic entity in a continuous and self-supervised manner. We propose a scene parsing 4D label map tracking to disentangle the spatial information explicitly, and a continuous deform module to disentangle the temporal motion implicitly. An object-aware volume rendering scheme is further introduced for the re-assembling of all the neural layers. We adopt a novel layered loss and motion-aware ray sampling strategy to enable efficient training for a large dynamic scene with multiple performers, Our framework further enables a variety of editing functions, i.e., manipulating the scale and location, duplicating or retiming individual neural layers to create numerous visual effects while preserving high realism. Extensive experiments demonstrate the effectiveness of our approach to achieve high-quality, photo-realistic, and editable free-viewpoint video generation for dynamic scenes.