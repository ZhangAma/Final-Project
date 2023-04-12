
每周分类神经辐射场 - lighting ![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)
=====================================================================================================================================
## 按类别筛选: 
 [全部](../weekly_nerf_cn.md) | [动态](./dynamic.md) | [编辑](./editing.md) | [快速](./fast.md) | [泛化](./generalization.md) | [人体](./human.md) | [视频](./video.md) | [光照](./lighting.md) | [重建](./reconstruction.md) | [纹理](./texture.md) | [语义](./semantic.md) | [姿态-SLAM](./pose-slam.md) | [其他](./others.md) 
## Dec27 - Jan3, 2023
## Dec25 - Dec31, 2022
## Dec18 - Dec24, 2022
## Dec11 - Dec17, 2022
## Dec4 - Dec10, 2022
## Nov27 - Dec3, 2022
  - [光场的神经子空间, TVCG2022](https://ieeexplore.ieee.org/abstract/document/9968104) | [code]
    > 我们引入了一个框架，用于用神经子空间的新概念来紧凑地表示光场内容。 虽然最近提出的神经光场表示通过将光场编码到单个神经网络中实现了很好的压缩结果，但统一设计并未针对光场中展示的复合结构进行优化。 此外，将光场的每一部分编码到一个网络中对于需要快速传输和解码的应用来说并不理想。 我们认识到这个问题与子空间学习的联系。 我们提出了一种使用几个小型神经网络的方法，专门研究特定光场段的神经子空间。 此外，我们在这些小型网络中提出了一种自适应权重共享策略，提高了参数效率。 实际上，该策略通过利用神经网络的分层结构，能够以协调一致的方式跟踪附近神经子空间之间的相似性。 此外，我们开发了一种软分类技术来提高神经表征的颜色预测准确性。 我们的实验结果表明，我们的方法在各种光场场景上比以前的方法更好地重建了光场。 我们进一步展示了其在具有不规则视点布局和动态场景内容的编码光场上的成功部署。
  - [用于实时全局照明的高效光探测器, SIGGRAPH-Asia2022](https://dl.acm.org/doi/abs/10.1145/3550454.3555452) | [code]
    > 再现基于物理的全局照明 (GI) 效果一直是许多实时图形应用程序的长期需求。 为了实现这一目标，许多最近的引擎采用了在预计算阶段烘焙的某种形式的光探测器。 不幸的是，由于探针存储、表示或查询的限制，预计算探针产生的 GI 效果相当有限。 在本文中，我们提出了一种基于探针的 GI 渲染的新方法，该方法可以在复杂场景中生成广泛的 GI 效果，包括具有多次反弹的光泽反射。 我们工作背后的关键贡献包括基于梯度的搜索算法和神经图像重建方法。 搜索算法旨在将探针的内容重新投影到任何查询视点，而不会引入视差误差，并快速收敛到最优解。 基于专用神经网络和多个 G 缓冲区的神经图像重建方法试图从由于分辨率有限或（潜在的）探头采样率低而导致的低质量输入中恢复高质量图像。 这种神经方法使光探针的生成变得高效。 此外，采用时间重投影策略和时间损失来提高动画序列的时间稳定性。 由于基于梯度的搜索算法的快速收敛速度和神经网络的轻量级设计，即使对于高分辨率 (1920×1080) 输出，整个流水线也实时运行（>30 帧/秒）。 已经对多个复杂场景进行了广泛的实验，以证明我们的方法优于最先进的方法。
  - [NeuLighting：使用不受约束的照片集重新照明的自由视点户外场景的神经照明, SIGGRAPH-Asia2022](https://dl.acm.org/doi/abs/10.1145/3550469.3555384) | [code]
    > 我们提出了 NeuLighting，这是一个新的框架，用于从一组稀疏的、不受约束的野外照片集中重新照明自由视点户外场景。 我们的框架将所有场景组件表示为由 MLP 参数化的连续函数，这些函数将 3D 位置和照明条件作为输入和输出反射率以及必要的室外照明属性。 与通常利用具有可控且一致的室内照明的训练图像的对象级重新照明方法不同，我们专注于更具挑战性的室外情况，其中所有图像都是在任意未知照明下捕获的。 我们方法的关键包括将每幅图像的光照压缩为解缠结的潜在向量的神经光照表示，以及一种新的自由视点重新光照方案，该方案对图像间的任意光照变化具有鲁棒性。 光照表示具有压缩性，可以解释各种光照，并且可以很容易地输入到基于查询的 NeuLighting 框架中，从而能够在任何一种新型光照下进行高效的阴影效果评估。 此外，为了产生高质量的投射阴影，我们根据场景几何形状和太阳方向估计太阳能见度图以指示阴影区域。 由于灵活且可解释的神经照明表示，我们的系统支持使用许多不同的照明源进行户外重新照明，包括自然图像、环境地图和延时视频。 新视角和照明下的高保真渲染证明了我们的方法相对于最先进的重新照明解决方案的优越性。
## Nov20 - Nov26, 2022
  - [折射物体的神经辐射场采样, SIGGRAPH-Asia2022](https://arxiv.org/abs/2211.14799) | [***``[code]``***](https://github.com/alexkeroro86/SampleNeRFRO)
    > 最近，神经辐射场 (NeRF) 中的可微分体绘制得到了广泛的关注，其变体取得了许多令人印象深刻的结果。然而，现有的方法通常假设场景是一个均匀的体积，因此光线沿着直线路径投射。在这项工作中，场景是一个具有分段恒定折射率的异质体积，如果它与不同的折射率相交，路径将弯曲。对于折射物体的新视图合成，我们基于 NeRF 的框架旨在从具有折射物体轮廓的多视图姿势图像中优化有界体积和边界的辐射场。为了解决这个具有挑战性的问题，场景的折射率是从轮廓中重建的。给定折射率，我们扩展了 NeRF 中的分层和分层采样技术，以允许沿着由 Eikonal 方程跟踪的弯曲路径绘制样本。结果表明，我们的框架在数量和质量上都优于最先进的方法，在感知相似性度量上表现出更好的性能，并且在几个合成和真实场景的渲染质量上有明显改善。
## Nov13 - Nov19, 2022
## Nov6 - Nov12, 2022
## Oct30 - Nov5, 2022
## Oct23 - Oct29, 2022
## Oct16 - Oct22, 2022
## Oct9 - Oct15, 2022
  - [LB-NERF：用于透明介质的光弯曲神经辐射场, ICIP2022](https://ieeexplore.ieee.org/abstract/document/9897642) | [code]
    > 神经辐射场 (NeRFs) 已被提出作为新颖的视图合成方法，并且由于其多功能性已被用于解决各种问题。 NeRF 可以使用假设直线光路的神经渲染来表示 3D 空间中的颜色和密度。但是，场景中具有不同折射率的介质，例如透明介质，会引起光的折射，打破了光路直线的假设。因此，不能在多视图图像中一致地学习 NeRF。为了解决这个问题，本研究提出了一种方法，通过引入光折射效应作为与源自相机中心的直线的偏移量来学习跨多个视点的一致辐射场。实验结果定量和定性地验证了在考虑透明物体的折射时，我们的方法可以比传统的 NeRF 方法更好地插入视点。
  - [IBL-NeRF：基于图像的神经辐射场照明公式](https://arxiv.org/abs/2210.08202) | [code]
    > 我们提出了 IBL-NeRF，它将大规模室内场景的神经辐射场 (NeRF) 分解为内在成分。以前的 NeRF 逆向渲染方法转换隐式体积以适应显式几何的渲染管道，并使用环境照明近似分割、孤立对象的视图。相比之下，我们的逆渲染扩展了原始的 NeRF 公式，以捕捉场景体积内照明的空间变化，以及表面属性。具体来说，将不同材质的场景分解为基于图像的渲染的内在组件，即反照率、粗糙度、表面法线、辐照度和预过滤辐射度。所有组件都被推断为来自 MLP 的神经图像，可以对大规模的一般场景进行建模。通过采用基于图像的 NeRF 公式，我们的方法继承了合成图像的卓越视觉质量和多视图一致性。我们展示了在具有复杂对象布局和灯光配置的场景上的性能，这些在以前的任何作品中都无法处理。
  - [使用树结构从辐射场估计神经反射场](https://arxiv.org/abs/2210.04217) | [code]
    > 我们提出了一种新方法，用于在未知光照下从一组姿势多视图图像中估计对象的神经反射场 (NReF)。 NReF 以分离的方式表示对象的 3D 几何和外观，并且很难仅从图像中估计。我们的方法通过利用神经辐射场（NeRF）作为代理表示来解决这个问题，我们从中进行进一步的分解。高质量的 NeRF 分解依赖于良好的几何信息提取以及良好的先验项来正确解决不同组件之间的歧义。为了从辐射场中提取高质量的几何信息，我们重新设计了一种新的基于射线投射的表面点提取方法。为了有效地计算和应用先验项，我们将不同的先验项转换为从辐射场提取的表面上的不同类型的滤波操作。然后，我们采用两种类型的辅助数据结构，即高斯 KD-tree 和八叉树，以支持在训练期间快速查询表面点和高效计算表面过滤器。基于此，我们设计了一个多级分解优化流程，用于从神经辐射场估计神经反射场。大量实验表明，我们的方法在不同数据上优于其他最先进的方法，并且能够实现高质量的自由视图重新照明以及材料编辑任务。
## Oct2 - Oct8, 2022
## Sep25 - Oct1, 2022
  - [神经全局照明：动态区域光下的交互式间接照明预测, TVCG2022](https://ieeexplore.ieee.org/abstract/document/9904431) | [code]
    > 我们提出了神经全局照明，这是一种在具有动态视点和区域照明的静态场景中快速渲染全全局照明的新方法。我们方法的关键思想是利用深度渲染网络来模拟从每个着色点到全局照明的复杂映射。为了有效地学习映射，我们提出了一种对神经网络友好的输入表示，包括每个着色点的属性、视点信息和组合照明表示，该表示能够与紧凑的神经网络进行高质量的拟合。为了合成高频全局光照效果，我们通过位置编码将低维输入转换为高维空间，并将渲染网络建模为深度全连接网络。此外，我们将屏幕空间神经缓冲区提供给我们的渲染网络，以将屏幕空间中的对象之间的全局信息共享到每个着色点。我们已经证明了我们的神经全局照明方法可以渲染各种场景，这些场景表现出复杂的全频全局照明效果，例如多次反射光泽互反射、渗色和焦散。
## Sep18 - Sep24, 2022
## Sep11 - Sep17, 2022
  - [StructNeRF：具有结构提示的室内场景的神经辐射场](https://arxiv.org/abs/2209.05277) | [code]
    > 神经辐射场 (NeRF) 使用密集捕获的输入图像实现照片般逼真的视图合成。然而，在给定稀疏视图的情况下，NeRF 的几何形状受到极大限制，导致新视图合成质量显着下降。受自监督深度估计方法的启发，我们提出了 StructNeRF，这是一种针对具有稀疏输入的室内场景的新颖视图合成的解决方案。 StructNeRF 利用自然嵌入在多视图输入中的结构提示来处理 NeRF 中的无约束几何问题。具体来说，它分别处理纹理和非纹理区域：提出了一种基于块的多视图一致光度损失来约束纹理区域的几何形状；对于非纹理平面，我们明确将它们限制为 3D 一致平面。通过密集的自监督深度约束，我们的方法提高了 NeRF 的几何和视图合成性能，而无需对外部数据进行任何额外的训练。对几个真实世界数据集的广泛实验表明，StructNeRF 在数量和质量上都超过了用于室内场景的最先进的方法。
## Sep4 - Sep10, 2022
## Aug28 - Sep3, 2022
  - [跨光谱神经辐射场, 3DV2022](https://arxiv.org/abs/2209.00648) | [code]
    > 我们提出了 X-NeRF，这是一种基于神经辐射场公式的学习交叉光谱场景表示的新方法，该方法给定从具有不同光谱灵敏度的相机捕获的图像。 X-NeRF 在训练期间优化跨光谱的相机姿势，并利用归一化跨设备坐标 (NXDC) 从任意视点呈现不同模态的图像，这些图像对齐并具有相同的分辨率。对 16 个具有彩色、多光谱和红外图像的前向场景进行的实验证实了 X-NeRF 在建模交叉光谱场景表示方面的有效性。
## Aug21 - Aug27, 2022
## Aug14 - Aug20, 2022
  - [从全向图像中捕捉休闲室内 HDR 辐射](https://arxiv.org/abs/2208.07903) | [code]
    > 我们提出了 PanoHDR-NeRF，这是一种新颖的管道，可以随意捕获大型室内场景的合理全 HDR 辐射场，而无需精心设置或复杂的捕获协议。首先，用户通过在场景周围自由挥动现成的相机来捕捉场景的低动态范围 (LDR) 全向视频。 然后，LDR2HDR 网络将捕获的 LDR 帧提升为 HDR，随后用于训练定制的 NeRF++ 模型。 由此产生的 PanoHDR-NeRF 管道可以从场景的任何位置估计完整的 HDR 全景图。 通过对各种真实场景的新测试数据集进行实验，在训练期间未看到的位置捕获地面实况 HDR 辐射，我们表明 PanoHDR-NeRF 可以预测来自任何场景点的合理辐射。我们还表明，由 PanoHDR-NeRF 生成的 HDR 图像可以合成正确的照明效果，从而能够使用正确照明的合成对象来增强室内场景。
  - [HDR-Plenoxels：自校准高动态范围辐射场, ECCV2022](https://arxiv.org/abs/2208.06787) | [code]
    > 我们提出了高动态范围辐射 (HDR) 场 HDR-Plenoxels，它学习 3D HDR 辐射场、几何信息和 2D 低动态范围 (LDR) 图像中固有的不同相机设置的全光函数。我们基于体素的体素渲染管道仅使用从不同相机设置中以端到端方式拍摄的多视图 LDR 图像来重建 HDR 辐射场，并且具有快速的收敛速度。为了处理现实世界场景中的各种相机，我们引入了一个色调映射模块，该模块对相机内的数字成像管道 (ISP) 进行建模并解开辐射设置。我们的色调映射模块允许我们通过控制每个新视图的辐射设置来进行渲染。最后，我们构建了一个具有不同相机条件的多视图数据集，这符合我们的问题设置。我们的实验表明，HDR-Plenoxels 可以仅从带有各种相机的 LDR 图像中表达细节和高质量的 HDR 新颖视图。
## Aug7 - Aug13, 2022
## Jul31 - Aug6, 2022
## Jul24 - Jul30, 2022
  - [具有全局照明的可重新照明的新视图合成的神经辐射转移场](https://arxiv.org/abs/2207.13607) | [code]
    > 给定场景的一组图像，从新颖的视图和光照条件重新渲染该场景是计算机视觉和图形学中一个重要且具有挑战性的问题。一方面，计算机视觉中的大多数现有作品通常对图像形成过程施加许多假设，例如直接照明和预定义的材料，使场景参数估计易于处理。另一方面，成熟的计算机图形学工具允许在给定所有场景参数的情况下对复杂的照片般逼真的光传输进行建模。结合这些方法，我们提出了一种通过学习神经预计算辐射传递函数来在新视图下重新点亮场景的方法，该函数使用新的环境图隐式处理全局光照效果。我们的方法可以在单一未知照明条件下对一组场景的真实图像进行单独监督。为了在训练期间消除任务的歧义，我们在训练过程中紧密集成了一个可微的路径跟踪器，并提出了合成 OLAT 和真实图像损失的组合。结果表明，与当前技术水平相比，场景参数的恢复解缠结得到了显着改善，因此，我们的重新渲染结果也更加真实和准确。
## Previous weeks
  - [野外的 NeRF：无约束照片集的神经辐射场, CVPR2021](https://arxiv.org/abs/2008.02268) | [code]
    > 我们提出了一种基于学习的方法，用于仅使用野外照片的非结构化集合来合成复杂场景的新视图。我们建立在神经辐射场 (NeRF) 的基础上，它使用多层感知器的权重将场景的密度和颜色建模为 3D 坐标的函数。虽然 NeRF 在受控设置下捕获的静态对象的图像上效果很好，但它无法在不受控的图像中模拟许多普遍存在的真实世界现象，例如可变照明或瞬态遮挡物。我们为 NeRF 引入了一系列扩展来解决这些问题，从而能够从互联网上获取的非结构化图像集合中进行准确的重建。我们将我们的系统（称为 NeRF-W）应用于著名地标的互联网照片集，并展示时间一致的新颖视图渲染，这些渲染比现有技术更接近真实感。
  - [Ha-NeRF：野外的幻觉神经辐射场, CVPR2022](https://rover-xingyu.github.io/Ha-NeRF/) | [***``[code]``***](https://github.com/rover-xingyu/Ha-NeRF)
    > 神经辐射场 (NeRF) 最近因其令人印象深刻的新颖视图合成能力而广受欢迎。本文研究了幻觉 NeRF 的问题：即在一天中的不同时间从一组旅游图像中恢复一个真实的 NeRF。现有的解决方案采用具有可控外观嵌入的 NeRF 在各种条件下渲染新颖的视图，但它们无法渲染具有看不见的外观的视图一致图像。为了解决这个问题，我们提出了一个用于构建幻觉 NeRF 的端到端框架，称为 Ha-NeRF。具体来说，我们提出了一个外观幻觉模块来处理随时间变化的外观并将它们转移到新的视图中。考虑到旅游图像的复杂遮挡，我们引入了一个反遮挡模块来准确地分解静态主体以获得可见性。合成数据和真实旅游照片集的实验结果表明，我们的方法可以产生幻觉，并从不同的视图呈现无遮挡的图像。
  - [黑暗中的 NeRF：来自嘈杂原始图像的高动态范围视图合成, CVPR2022(oral)](https://bmild.github.io/rawnerf/) | [***``[code]``***](https://github.com/google-research/multinerf)
    > 神经辐射场 (NeRF) 是一种从姿势输入图像的集合中合成高质量新颖视图的技术。与大多数视图合成方法一样，NeRF 使用色调映射低动态范围（LDR）作为输入；这些图像已由有损相机管道处理，该管道可以平滑细节、剪辑高光并扭曲原始传感器数据的简单噪声分布。我们将 NeRF 修改为直接在线性原始图像上进行训练，保留场景的完整动态范围。通过从生成的 NeRF 渲染原始输出图像，我们可以执行新颖的高动态范围 (HDR) 视图合成任务。除了改变相机视角之外，我们还可以在事后操纵焦点、曝光和色调映射。尽管单个原始图像看起来比后处理的图像噪声大得多，但我们表明 NeRF 对原始噪声的零均值分布具有高度鲁棒性。当针对许多嘈杂的原始输入 (25-200) 进行优化时，NeRF 生成的场景表示非常准确，以至于其渲染的新颖视图优于在相同宽基线输入图像上运行的专用单图像和多图像深度原始降噪器。因此，我们的方法（我们称为 RawNeRF）可以从在近黑暗中捕获的极其嘈杂的图像中重建场景。
  - [NeRV：用于重新照明和视图合成的神经反射率和可见性场, CVPR2021](https://pratulsrinivasan.github.io/nerv/) | [code]
    > 我们提出了一种方法，该方法将由不受约束的已知照明照明的场景的一组图像作为输入，并生成可以在任意照明条件下从新视点渲染的 3D 表示作为输出。我们的方法将场景表示为参数化为 MLP 的连续体积函数，其输入是 3D 位置，其输出是该输入位置的以下场景属性：体积密度、表面法线、材料参数、到任何方向上第一个表面交点的距离，以及任何方向的外部环境的可见性。总之，这些允许我们在任意照明下渲染物体的新视图，包括间接照明效果。预测的能见度和表面相交场对于我们的模型在训练期间模拟直接和间接照明的能力至关重要，因为先前工作使用的蛮力技术对于具有单灯的受控设置之外的照明条件是难以处理的。我们的方法在恢复可重新照明的 3D 场景表示方面优于替代方法，并且在对先前工作构成重大挑战的复杂照明设置中表现良好。
  - [NeX：具有神经基础扩展的实时视图合成, CVPR2021(oral)](https://nex-mpi.github.io/) | [***``[code]``***](https://github.com/nex-mpi/nex-code/)
    > 我们提出了 NeX，这是一种基于多平面图像 (MPI) 增强的新型视图合成的新方法，可以实时再现 NeXt 级别的视图相关效果。与使用一组简单 RGBα 平面的传统 MPI 不同，我们的技术通过将每个像素参数化为从神经网络学习的基函数的线性组合来模拟视图相关的效果。此外，我们提出了一种混合隐式-显式建模策略，该策略改进了精细细节并产生了最先进的结果。我们的方法在基准前向数据集以及我们新引入的数据集上进行了评估，该数据集旨在测试与视图相关的建模的极限，具有明显更具挑战性的效果，例如 CD 上的彩虹反射。我们的方法在这些数据集的所有主要指标上都取得了最好的总体得分，渲染时间比现有技术快 1000 倍以上。
  - [NeRFactor：未知光照下形状和反射率的神经分解, TOG 2021 (Proc. SIGGRAPH Asia)](https://xiuming.info/projects/nerfactor/) | [code]
    > 我们解决了从由一种未知光照条件照射的物体的多视图图像（及其相机姿势）中恢复物体的形状和空间变化反射率的问题。这使得能够在任意环境照明下渲染对象的新颖视图并编辑对象的材质属性。我们方法的关键，我们称之为神经辐射分解（NeRFactor），是提取神经辐射场（NeRF）的体积几何[Mildenhall et al。 2020] 将对象表示为表面表示，然后在解决空间变化的反射率和环境照明的同时联合细化几何。具体来说，NeRFactor 在没有任何监督的情况下恢复表面法线、光能见度、反照率和双向反射分布函数 (BRDF) 的 3D 神经场，仅使用重新渲染损失、简单的平滑先验和从真实数据中学习的数据驱动的 BRDF 先验-世界BRDF测量。通过显式建模光可见性，NeRFactor 能够从反照率中分离出阴影，并在任意光照条件下合成逼真的软阴影或硬阴影。 NeRFactor 能够恢复令人信服的 3D 模型，用于在合成场景和真实场景的这种具有挑战性且约束不足的捕获设置中进行自由视点重新照明。定性和定量实验表明，NeRFactor 在各种任务中都优于经典和基于深度学习的最新技术。我们的视频、代码和数据可在 people.csail.mit.edu/xiuming/projects/nerfactor/ 上找到。
  - [Fig-NeRF：用于 3D 对象类别建模的图地面神经辐射场, 3DV2021](https://fig-nerf.github.io/) | [code]
    > 我们研究使用神经辐射场 (NeRF) 从输入图像的集合中学习高质量的 3D 对象类别模型。与以前的工作相比，我们能够做到这一点，同时将前景对象与不同的背景分开。我们通过 2 分量 NeRF 模型 FiG-NeRF 实现了这一点，该模型更喜欢将场景解释为几何恒定的背景和代表对象类别的可变形前景。我们表明，这种方法可以仅使用光度监督和随意捕获的对象图像来学习准确的 3D 对象类别模型。此外，我们的两部分分解允许模型执行准确和清晰的模态分割。我们使用合成的、实验室捕获的和野外数据，通过视图合成和图像保真度指标对我们的方法进行定量评估。我们的结果证明了令人信服的 3D 对象类别建模，其性能超过了现有方法的性能。
  - [NerfingMVS：室内多视角立体神经辐射场的引导优化, ICCV2021(oral)](https://arxiv.org/abs/2109.01129) | [***``[code]``***](https://github.com/weiyithu/NerfingMVS)
    > 在这项工作中，我们提出了一种新的多视图深度估计方法，该方法在最近提出的神经辐射场 (NeRF) 上利用了传统的 SfM 重建和基于学习的先验。与现有的依赖于估计对应的基于神经网络的优化方法不同，我们的方法直接优化隐式体积，消除了在室内场景中匹配像素的挑战性步骤。我们方法的关键是利用基于学习的先验来指导 NeRF 的优化过程。我们的系统首先通过微调其稀疏 SfM 重建来适应目标场景上的单目深度网络。然后，我们证明了 NeRF 的形状-辐射模糊性仍然存在于室内环境中，并建议通过采用适应的深度先验来监控体绘制的采样过程来解决这个问题。最后，通过对渲染图像进行误差计算获得的每像素置信度图可用于进一步提高深度质量。实验表明，我们提出的框架在室内场景中显着优于最先进的方法，在基于对应的优化和基于 NeRF 的优化对适应深度先验的有效性方面提出了令人惊讶的发现。此外，我们表明引导优化方案不会牺牲神经辐射场的原始合成能力，提高了可见视图和新视图的渲染质量。