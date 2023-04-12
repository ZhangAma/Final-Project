
Weekly Classified Neural Radiance Fields - lighting ![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)
====================================================================================================================================================================
## Filter by classes: 
 [all](../weekly_nerf.md) | [dynamic](./dynamic.md) | [editing](./editing.md) | [fast](./fast.md) | [generalization](./generalization.md) | [human](./human.md) | [video](./video.md) | [lighting](./lighting.md) | [reconstruction](./reconstruction.md) | [texture](./texture.md) | [semantic](./semantic.md) | [pose-slam](./pose-slam.md) | [others](./others.md) 
## Dec27 - Jan3, 2023
## Dec25 - Dec31, 2022
## Dec18 - Dec24, 2022
## Dec11 - Dec17, 2022
## Dec4 - Dec10, 2022
## Nov27 - Dec3, 2022
  - [Neural Subspaces for Light Fields, TVCG2022](https://ieeexplore.ieee.org/abstract/document/9968104) | [code]
    > We introduce a framework for compactly representing light field content with the novel concept of neural subspaces. While the recently proposed neural light field representation achieves great compression results by encoding a light field into a single neural network, the unified design is not optimized for the composite structures exhibited in light fields. Moreover, encoding every part of the light field into one network is not ideal for applications that require rapid transmission and decoding. We recognize this problem's connection to subspace learning. We present a method that uses several small neural networks, specializing in learning the neural subspace for a particular light field segment. Moreover, we propose an adaptive weight sharing strategy among those small networks, improving parameter efficiency. In effect, this strategy enables a concerted way to track the similarity among nearby neural subspaces by leveraging the layered structure of neural networks. Furthermore, we develop a soft-classification technique to enhance the color prediction accuracy of neural representations. Our experimental results show that our method better reconstructs the light field than previous methods on various light field scenes. We further demonstrate its successful deployment on encoding light fields with irregular viewpoint layout and dynamic scene content.
  - [Efficient Light Probes for Real-Time Global Illumination, SIGGRAPH-Asia2022](https://dl.acm.org/doi/abs/10.1145/3550454.3555452) | [code]
    > Reproducing physically-based global illumination (GI) effects has been a long-standing demand for many real-time graphical applications. In pursuit of this goal, many recent engines resort to some form of light probes baked in a precomputation stage. Unfortunately, the GI effects stemming from the precomputed probes are rather limited due to the constraints in the probe storage, representation or query. In this paper, we propose a new method for probe-based GI rendering which can generate a wide range of GI effects, including glossy reflection with multiple bounces, in complex scenes. The key contributions behind our work include a gradient-based search algorithm and a neural image reconstruction method. The search algorithm is designed to reproject the probes' contents to any query viewpoint, without introducing parallax errors, and converges fast to the optimal solution. The neural image reconstruction method, based on a dedicated neural network and several G-buffers, tries to recover high-quality images from low-quality inputs due to limited resolution or (potential) low sampling rate of the probes. This neural method makes the generation of light probes efficient. Moreover, a temporal reprojection strategy and a temporal loss are employed to improve temporal stability for animation sequences. The whole pipeline runs in realtime (>30 frames per second) even for high-resolution (1920×1080) outputs, thanks to the fast convergence rate of the gradient-based search algorithm and a light-weight design of the neural network. Extensive experiments on multiple complex scenes have been conducted to show the superiority of our method over the state-of-the-arts.
  - [NeuLighting: Neural Lighting for Free Viewpoint Outdoor Scene Relighting with Unconstrained Photo Collections, SIGGRAPH-Asia2022](https://dl.acm.org/doi/abs/10.1145/3550469.3555384) | [code]
    > We propose NeuLighting, a new framework for free viewpoint outdoor scene relighting from a sparse set of unconstrained in-the-wild photo collections. Our framework represents all the scene components as continuous functions parameterized by MLPs that take a 3D location and the lighting condition as input and output reflectance and necessary outdoor illumination properties. Unlike object-level relighting methods which often leverage training images with controllable and consistent indoor illumination, we concentrate on the more challenging outdoor situation where all the images are captured under arbitrary unknown illumination. The key to our method includes a neural lighting representation that compresses the per-image illumination into a disentangled latent vector, and a new free viewpoint relighting scheme that is robust to arbitrary lighting variations across images. The lighting representation is compressive to explain a wide range of illumination and can be easily fed into the query-based NeuLighting framework, enabling efficient shading effect evaluation under any kind of novel illumination. Furthermore, to produce high-quality cast shadows, we estimate the sun visibility map to indicate the shadow regions according to the scene geometry and the sun direction. Thanks to the flexible and explainable neural lighting representation, our system supports outdoor relighting with many different illumination sources, including natural images, environment maps, and time-lapse videos. The high-fidelity renderings under novel views and illumination prove the superiority of our method against state-of-the-art relighting solutions.
## Nov20 - Nov26, 2022
  - [Sampling Neural Radiance Fields for Refractive Objects, SIGGRAPH-Asia2022](https://arxiv.org/abs/2211.14799) | [***``[code]``***](https://github.com/alexkeroro86/SampleNeRFRO)
    > Recently, differentiable volume rendering in neural radiance fields (NeRF) has gained a lot of popularity, and its variants have attained many impressive results. However, existing methods usually assume the scene is a homogeneous volume so that a ray is cast along the straight path. In this work, the scene is instead a heterogeneous volume with a piecewise-constant refractive index, where the path will be curved if it intersects the different refractive indices. For novel view synthesis of refractive objects, our NeRF-based framework aims to optimize the radiance fields of bounded volume and boundary from multi-view posed images with refractive object silhouettes. To tackle this challenging problem, the refractive index of a scene is reconstructed from silhouettes. Given the refractive index, we extend the stratified and hierarchical sampling techniques in NeRF to allow drawing samples along a curved path tracked by the Eikonal equation. The results indicate that our framework outperforms the state-of-the-art method both quantitatively and qualitatively, demonstrating better performance on the perceptual similarity metric and an apparent improvement in the rendering quality on several synthetic and real scenes.
## Nov13 - Nov19, 2022
## Nov6 - Nov12, 2022
## Oct30 - Nov5, 2022
## Oct23 - Oct29, 2022
## Oct16 - Oct22, 2022
## Oct9 - Oct15, 2022
  - [LB-NERF: Light Bending Neural Radiance Fields for Transparent Medium, ICIP2022](https://ieeexplore.ieee.org/abstract/document/9897642) | [code]
    > Neural radiance fields (NeRFs) have been proposed as methods of novel view synthesis and have been used to address various problems because of its versatility. NeRF can represent colors and densities in 3D space using neural rendering assuming a straight light path. However, a medium with a different refractive index in the scene, such as a transparent medium, causes light refraction and breaks the assumption of the straight path of light. Therefore, the NeRFs cannot be learned consistently across multi-view images. To solve this problem, this study proposes a method to learn consistent radiance fields across multiple viewpoints by introducing the light refraction effect as an offset from the straight line originating from the camera center. The experimental results quantitatively and qualitatively verified that our method can interpolate viewpoints better than the conventional NeRF method when considering the refraction of transparent objects.
  - [IBL-NeRF: Image-Based Lighting Formulation of Neural Radiance Fields](https://arxiv.org/abs/2210.08202) | [code]
    > We propose IBL-NeRF, which decomposes the neural radiance fields (NeRF) of large-scale indoor scenes into intrinsic components. Previous approaches for the inverse rendering of NeRF transform the implicit volume to fit the rendering pipeline of explicit geometry, and approximate the views of segmented, isolated objects with environment lighting. In contrast, our inverse rendering extends the original NeRF formulation to capture the spatial variation of lighting within the scene volume, in addition to surface properties. Specifically, the scenes of diverse materials are decomposed into intrinsic components for image-based rendering, namely, albedo, roughness, surface normal, irradiance, and prefiltered radiance. All of the components are inferred as neural images from MLP, which can model large-scale general scenes. By adopting the image-based formulation of NeRF, our approach inherits superior visual quality and multi-view consistency for synthesized images. We demonstrate the performance on scenes with complex object layouts and light configurations, which could not be processed in any of the previous works.
  - [Estimating Neural Reflectance Field from Radiance Field using Tree Structures](https://arxiv.org/abs/2210.04217) | [code]
    > We present a new method for estimating the Neural Reflectance Field (NReF) of an object from a set of posed multi-view images under unknown lighting. NReF represents 3D geometry and appearance of objects in a disentangled manner, and are hard to be estimated from images only. Our method solves this problem by exploiting the Neural Radiance Field (NeRF) as a proxy representation, from which we perform further decomposition. A high-quality NeRF decomposition relies on good geometry information extraction as well as good prior terms to properly resolve ambiguities between different components. To extract high-quality geometry information from radiance fields, we re-design a new ray-casting based method for surface point extraction. To efficiently compute and apply prior terms, we convert different prior terms into different type of filter operations on the surface extracted from radiance field. We then employ two type of auxiliary data structures, namely Gaussian KD-tree and octree, to support fast querying of surface points and efficient computation of surface filters during training. Based on this, we design a multi-stage decomposition optimization pipeline for estimating neural reflectance field from neural radiance fields. Extensive experiments show our method outperforms other state-of-the-art methods on different data, and enable high-quality free-view relighting as well as material editing tasks.
## Oct2 - Oct8, 2022
## Sep25 - Oct1, 2022
  - [Neural Global Illumination: Interactive Indirect Illumination Prediction under Dynamic Area Lights, TVCG2022](https://ieeexplore.ieee.org/abstract/document/9904431) | [code]
    > We propose neural global illumination, a novel method for fast rendering full global illumination in static scenes with dynamic viewpoint and area lighting. The key idea of our method is to utilize a deep rendering network to model the complex mapping from each shading point to global illumination. To efficiently learn the mapping, we propose a neural-network-friendly input representation including attributes of each shading point, viewpoint information, and a combinational lighting representation that enables high-quality fitting with a compact neural network. To synthesize high-frequency global illumination effects, we transform the low-dimension input to higher-dimension space by positional encoding and model the rendering network as a deep fully-connected network. Besides, we feed a screen-space neural buffer to our rendering network to share global information between objects in the screen-space to each shading point. We have demonstrated our neural global illumination method in rendering a wide variety of scenes exhibiting complex and all-frequency global illumination effects such as multiple-bounce glossy interreflection, color bleeding, and caustics.
## Sep18 - Sep24, 2022
## Sep11 - Sep17, 2022
  - [StructNeRF: Neural Radiance Fields for Indoor Scenes with Structural Hints](https://arxiv.org/abs/2209.05277) | [code]
    > Neural Radiance Fields (NeRF) achieve photo-realistic view synthesis with densely captured input images. However, the geometry of NeRF is extremely under-constrained given sparse views, resulting in significant degradation of novel view synthesis quality. Inspired by self-supervised depth estimation methods, we propose StructNeRF, a solution to novel view synthesis for indoor scenes with sparse inputs. StructNeRF leverages the structural hints naturally embedded in multi-view inputs to handle the unconstrained geometry issue in NeRF. Specifically, it tackles the texture and non-texture regions respectively: a patch-based multi-view consistent photometric loss is proposed to constrain the geometry of textured regions; for non-textured ones, we explicitly restrict them to be 3D consistent planes. Through the dense self-supervised depth constraints, our method improves both the geometry and the view synthesis performance of NeRF without any additional training on external data. Extensive experiments on several real-world datasets demonstrate that StructNeRF surpasses state-of-the-art methods for indoor scenes with sparse inputs both quantitatively and qualitatively.
## Sep4 - Sep10, 2022
## Aug28 - Sep3, 2022
  - [Cross-Spectral Neural Radiance Fields, 3DV2022](https://arxiv.org/abs/2209.00648) | [code]
    > We propose X-NeRF, a novel method to learn a Cross-Spectral scene representation given images captured from cameras with different light spectrum sensitivity, based on the Neural Radiance Fields formulation. X-NeRF optimizes camera poses across spectra during training and exploits Normalized Cross-Device Coordinates (NXDC) to render images of different modalities from arbitrary viewpoints, which are aligned and at the same resolution. Experiments on 16 forward-facing scenes, featuring color, multi-spectral and infrared images, confirm the effectiveness of X-NeRF at modeling Cross-Spectral scene representations.
## Aug21 - Aug27, 2022
## Aug14 - Aug20, 2022
  - [Casual Indoor HDR Radiance Capture from Omnidirectional Images](https://arxiv.org/abs/2208.07903) | [code]
    > We present PanoHDR-NeRF, a novel pipeline to casually capture a plausible full HDR radiance field of a large indoor scene without elaborate setups or complex capture protocols. First, a user captures a low dynamic range (LDR) omnidirectional video of the scene by freely waving an off-the-shelf camera around the scene. Then, an LDR2HDR network uplifts the captured LDR frames to HDR, subsequently used to train a tailored NeRF++ model. The resulting PanoHDR-NeRF pipeline can estimate full HDR panoramas from any location of the scene. Through experiments on a novel test dataset of a variety of real scenes with the ground truth HDR radiance captured at locations not seen during training, we show that PanoHDR-NeRF predicts plausible radiance from any scene point. We also show that the HDR images produced by PanoHDR-NeRF can synthesize correct lighting effects, enabling the augmentation of indoor scenes with synthetic objects that are lit correctly.
  - [HDR-Plenoxels: Self-Calibrating High Dynamic Range Radiance Fields, ECCV2022](https://arxiv.org/abs/2208.06787) | [code]
    > We propose high dynamic range radiance (HDR) fields, HDR-Plenoxels, that learn a plenoptic function of 3D HDR radiance fields, geometry information, and varying camera settings inherent in 2D low dynamic range (LDR) images. Our voxel-based volume rendering pipeline reconstructs HDR radiance fields with only multi-view LDR images taken from varying camera settings in an end-to-end manner and has a fast convergence speed. To deal with various cameras in real-world scenarios, we introduce a tone mapping module that models the digital in-camera imaging pipeline (ISP) and disentangles radiometric settings. Our tone mapping module allows us to render by controlling the radiometric settings of each novel view. Finally, we build a multi-view dataset with varying camera conditions, which fits our problem setting. Our experiments show that HDR-Plenoxels can express detail and high-quality HDR novel views from only LDR images with various cameras.
## Aug7 - Aug13, 2022
## Jul31 - Aug6, 2022
## Jul24 - Jul30, 2022
  - [Neural Radiance Transfer Fields for Relightable Novel-view Synthesis with Global Illumination](https://arxiv.org/abs/2207.13607) | [code]
    > Given a set of images of a scene, the re-rendering of this scene from novel views and lighting conditions is an important and challenging problem in Computer Vision and Graphics. On the one hand, most existing works in Computer Vision usually impose many assumptions regarding the image formation process, e.g. direct illumination and predefined materials, to make scene parameter estimation tractable. On the other hand, mature Computer Graphics tools allow modeling of complex photo-realistic light transport given all the scene parameters. Combining these approaches, we propose a method for scene relighting under novel views by learning a neural precomputed radiance transfer function, which implicitly handles global illumination effects using novel environment maps. Our method can be solely supervised on a set of real images of the scene under a single unknown lighting condition. To disambiguate the task during training, we tightly integrate a differentiable path tracer in the training process and propose a combination of a synthesized OLAT and a real image loss. Results show that the recovered disentanglement of scene parameters improves significantly over the current state of the art and, thus, also our re-rendering results are more realistic and accurate.
## Previous weeks
  - [NeRF in the Wild: Neural Radiance Fields for Unconstrained Photo Collections, CVPR2021](https://arxiv.org/abs/2008.02268) | [code]
    > We present a learning-based method for synthesizing novel views of complex scenes using only unstructured collections of in-the-wild photographs. We build on Neural Radiance Fields (NeRF), which uses the weights of a multilayer perceptron to model the density and color of a scene as a function of 3D coordinates. While NeRF works well on images of static subjects captured under controlled settings, it is incapable of modeling many ubiquitous, real-world phenomena in uncontrolled images, such as variable illumination or transient occluders. We introduce a series of extensions to NeRF to address these issues, thereby enabling accurate reconstructions from unstructured image collections taken from the internet. We apply our system, dubbed NeRF-W, to internet photo collections of famous landmarks, and demonstrate temporally consistent novel view renderings that are significantly closer to photorealism than the prior state of the art.
  - [Ha-NeRF: Hallucinated Neural Radiance Fields in the Wild, CVPR2022](https://rover-xingyu.github.io/Ha-NeRF/) | [***``[code]``***](https://github.com/rover-xingyu/Ha-NeRF)
    > Neural Radiance Fields (NeRF) has recently gained popularity for its impressive novel view synthesis ability. This paper studies the problem of hallucinated NeRF: i.e., recovering a realistic NeRF at a different time of day from a group of tourism images. Existing solutions adopt NeRF with a controllable appearance embedding to render novel views under various conditions, but they cannot render view-consistent images with an unseen appearance. To solve this problem, we present an end-to-end framework for constructing a hallucinated NeRF, dubbed as Ha-NeRF. Specifically, we propose an appearance hallucination module to handle time-varying appearances and transfer them to novel views. Considering the complex occlusions of tourism images, we introduce an anti-occlusion module to decompose the static subjects for visibility accurately. Experimental results on synthetic data and real tourism photo collections demonstrate that our method can hallucinate the desired appearances and render occlusion-free images from different views.
  - [NeRF in the Dark: High Dynamic Range View Synthesis from Noisy Raw Images, CVPR2022(oral)](https://bmild.github.io/rawnerf/) | [***``[code]``***](https://github.com/google-research/multinerf)
    > Neural Radiance Fields (NeRF) is a technique for high quality novel view synthesis from a collection of posed input images. Like most view synthesis methods, NeRF uses tonemapped low dynamic range (LDR) as input; these images have been processed by a lossy camera pipeline that smooths detail, clips highlights, and distorts the simple noise distribution of raw sensor data. We modify NeRF to instead train directly on linear raw images, preserving the scene's full dynamic range. By rendering raw output images from the resulting NeRF, we can perform novel high dynamic range (HDR) view synthesis tasks. In addition to changing the camera viewpoint, we can manipulate focus, exposure, and tonemapping after the fact. Although a single raw image appears significantly more noisy than a postprocessed one, we show that NeRF is highly robust to the zero-mean distribution of raw noise. When optimized over many noisy raw inputs (25-200), NeRF produces a scene representation so accurate that its rendered novel views outperform dedicated single and multi-image deep raw denoisers run on the same wide baseline input images. As a result, our method, which we call RawNeRF, can reconstruct scenes from extremely noisy images captured in near-darkness.
  - [NeRV: Neural Reflectance and Visibility Fields for Relighting and View Synthesis, CVPR2021](https://pratulsrinivasan.github.io/nerv/) | [code]
    > We present a method that takes as input a set of images of a scene illuminated by unconstrained known lighting, and produces as output a 3D representation that can be rendered from novel viewpoints under arbitrary lighting conditions. Our method represents the scene as a continuous volumetric function parameterized as MLPs whose inputs are a 3D location and whose outputs are the following scene properties at that input location: volume density, surface normal, material parameters, distance to the first surface intersection in any direction, and visibility of the external environment in any direction. Together, these allow us to render novel views of the object under arbitrary lighting, including indirect illumination effects. The predicted visibility and surface intersection fields are critical to our model's ability to simulate direct and indirect illumination during training, because the brute-force techniques used by prior work are intractable for lighting conditions outside of controlled setups with a single light. Our method outperforms alternative approaches for recovering relightable 3D scene representations, and performs well in complex lighting settings that have posed a significant challenge to prior work.
  - [NeX: Real-time View Synthesis with Neural Basis Expansion, CVPR2021(oral)](https://nex-mpi.github.io/) | [***``[code]``***](https://github.com/nex-mpi/nex-code/)
    > We present NeX, a new approach to novel view synthesis based on enhancements of multiplane image (MPI) that can reproduce NeXt-level view-dependent effects---in real time. Unlike traditional MPI that uses a set of simple RGBα planes, our technique models view-dependent effects by instead parameterizing each pixel as a linear combination of basis functions learned from a neural network. Moreover, we propose a hybrid implicit-explicit modeling strategy that improves upon fine detail and produces state-of-the-art results. Our method is evaluated on benchmark forward-facing datasets as well as our newly-introduced dataset designed to test the limit of view-dependent modeling with significantly more challenging effects such as the rainbow reflections on a CD. Our method achieves the best overall scores across all major metrics on these datasets with more than 1000× faster rendering time than the state of the art.
  - [NeRFactor: Neural Factorization of Shape and Reflectance Under an Unknown Illumination, TOG 2021 (Proc. SIGGRAPH Asia)](https://xiuming.info/projects/nerfactor/) | [code]
    > We address the problem of recovering the shape and spatially-varying reflectance of an object from multi-view images (and their camera poses) of an object illuminated by one unknown lighting condition. This enables the rendering of novel views of the object under arbitrary environment lighting and editing of the object's material properties. The key to our approach, which we call Neural Radiance Factorization (NeRFactor), is to distill the volumetric geometry of a Neural Radiance Field (NeRF) [Mildenhall et al. 2020] representation of the object into a surface representation and then jointly refine the geometry while solving for the spatially-varying reflectance and environment lighting. Specifically, NeRFactor recovers 3D neural fields of surface normals, light visibility, albedo, and Bidirectional Reflectance Distribution Functions (BRDFs) without any supervision, using only a re-rendering loss, simple smoothness priors, and a data-driven BRDF prior learned from real-world BRDF measurements. By explicitly modeling light visibility, NeRFactor is able to separate shadows from albedo and synthesize realistic soft or hard shadows under arbitrary lighting conditions. NeRFactor is able to recover convincing 3D models for free-viewpoint relighting in this challenging and underconstrained capture setup for both synthetic and real scenes. Qualitative and quantitative experiments show that NeRFactor outperforms classic and deep learning-based state of the art across various tasks. Our videos, code, and data are available at people.csail.mit.edu/xiuming/projects/nerfactor/.
  - [FiG-NeRF: Figure Ground Neural Radiance Fields for 3D Object Category Modelling, 3DV2021](https://fig-nerf.github.io/) | [code]
    > We investigate the use of Neural Radiance Fields (NeRF) to learn high quality 3D object category models from collections of input images. In contrast to previous work, we are able to do this whilst simultaneously separating foreground objects from their varying backgrounds. We achieve this via a 2-component NeRF model, FiG-NeRF, that prefers explanation of the scene as a geometrically constant background and a deformable foreground that represents the object category. We show that this method can learn accurate 3D object category models using only photometric supervision and casually captured images of the objects. Additionally, our 2-part decomposition allows the model to perform accurate and crisp amodal segmentation. We quantitatively evaluate our method with view synthesis and image fidelity metrics, using synthetic, lab-captured, and in-the-wild data. Our results demonstrate convincing 3D object category modelling that exceed the performance of existing methods.
  - [NerfingMVS: Guided Optimization of Neural Radiance Fields for Indoor Multi-view Stereo, ICCV2021(oral)](https://arxiv.org/abs/2109.01129) | [***``[code]``***](https://github.com/weiyithu/NerfingMVS)
    > In this work, we present a new multi-view depth estimation method that utilizes both conventional SfM reconstruction and learning-based priors over the recently proposed neural radiance fields (NeRF). Unlike existing neural network based optimization method that relies on estimated correspondences, our method directly optimizes over implicit volumes, eliminating the challenging step of matching pixels in indoor scenes. The key to our approach is to utilize the learning-based priors to guide the optimization process of NeRF. Our system firstly adapts a monocular depth network over the target scene by finetuning on its sparse SfM reconstruction. Then, we show that the shape-radiance ambiguity of NeRF still exists in indoor environments and propose to address the issue by employing the adapted depth priors to monitor the sampling process of volume rendering. Finally, a per-pixel confidence map acquired by error computation on the rendered image can be used to further improve the depth quality. Experiments show that our proposed framework significantly outperforms state-of-the-art methods on indoor scenes, with surprising findings presented on the effectiveness of correspondence-based optimization and NeRF-based optimization over the adapted depth priors. In addition, we show that the guided optimization scheme does not sacrifice the original synthesis capability of neural radiance fields, improving the rendering quality on both seen and novel views.