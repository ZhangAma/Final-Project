
每周分类神经辐射场 - generalization ![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)
===========================================================================================================================================
## 按类别筛选: 
 [全部](../weekly_nerf_cn.md) | [动态](./dynamic.md) | [编辑](./editing.md) | [快速](./fast.md) | [泛化](./generalization.md) | [人体](./human.md) | [视频](./video.md) | [光照](./lighting.md) | [重建](./reconstruction.md) | [纹理](./texture.md) | [语义](./semantic.md) | [姿态-SLAM](./pose-slam.md) | [其他](./others.md) 
## Dec27 - Jan3, 2023
## Dec25 - Dec31, 2022
  - [用于高质量视图合成的稀疏 RGB-D 图像的神经辐射场, TPAMI2022](https://ieeexplore.ieee.org/abstract/document/9999509) | [code]
    > 最近提出的神经辐射场 (NeRF) 使用作为多层感知器 (MLP) 制定的连续函数来模拟 3D 场景的外观和几何形状。 这使得新视图的逼真合成成为可能，即使对于具有视图依赖外观的场景也是如此。 此后，许多后续工作以不同方式扩展了 NeRF。 然而，该方法的一个基本限制仍然是它需要从密集放置的视点捕获大量图像以进行高质量合成，并且当捕获的视图数量不足时，结果的质量会迅速下降。 为了解决这个问题，我们提出了一种新的基于 NeRF 的框架，该框架能够仅使用一组稀疏的 RGB-D 图像进行高质量的视图合成，这些图像可以在当前的消费设备上使用相机和 LiDAR 传感器轻松捕获。 首先，从捕获的 RGB-D 图像重建场景的几何代理。 然后可以使用重建场景的渲染以及精确的相机参数来预训练网络。 最后，使用少量真实捕获的图像对网络进行微调。 我们进一步引入了一个补丁鉴别器，以在微调期间在新颖的视图下监督网络，并在提高合成质量之前引入 3D 颜色。 我们证明了我们的方法可以从少至 6 个 RGB-D 图像生成 3D 场景的任意新颖视图。 大量实验表明，与现有的基于 NeRF 的方法相比，我们的方法有所改进，包括旨在减少输入图像数量的方法。
## Dec18 - Dec24, 2022
## Dec11 - Dec17, 2022
## Dec4 - Dec10, 2022
  - [图像生成器的扩散引导域自适应](https://arxiv.org/abs/2212.04473) | [code]
    > 能否将文本到图像扩散模型用作训练目标，让 GAN 生成器适应另一个领域？ 在本文中，我们展示了无分类器指导可以用作评论家，并使生成器能够从大规模文本到图像扩散模型中提取知识。 生成器可以有效地转移到文本提示指示的新域中，而无需访问目标域中的真实样本。 我们通过大量实验证明了我们方法的有效性和可控性。 尽管没有经过训练来最小化 CLIP 损失，但我们的模型在短提示上获得了同样高的 CLIP 分数和显着降低的 FID，并且在长而复杂的提示上在定性和定量上都优于基线。 据我们所知，所提出的方法是首次尝试将大规模预训练扩散模型和蒸馏采样结合起来用于文本驱动的图像生成器域自适应，并提供了以前无法实现的质量。 此外，我们将我们的工作扩展到基于 3D 风格的生成器和 DreamBooth 指南。
  - [NeRDi：以语言引导扩散作为一般图像先验的单视图 NeRF 合成](https://arxiv.org/abs/2212.03267) | [code]
    > 2D 到 3D 重建是一个病态问题，但由于人类多年来积累的 3D 世界先验知识，因此擅长解决这个问题。 受此观察的驱动，我们提出了 NeRDi，这是一种单视图 NeRF 合成框架，具有来自 2D 扩散模型的一般图像先验。 将单视图重建制定为图像条件 3D 生成问题，我们通过在输入视图约束下使用预训练图像扩散模型最小化其任意视图渲染上的扩散损失来优化 NeRF 表示。 我们利用现成的视觉语言模型，并引入两部分语言指导作为扩散模型的条件输入。 这本质上有助于提高多视图内容的一致性，因为它缩小了以单视图输入图像的语义和视觉特征为条件的一般图像先验范围。 此外，我们引入了基于估计深度图的几何损失，以正则化 NeRF 的底层 3D 几何。 DTU MVS 数据集上的实验结果表明，与在此数据集上训练的现有方法相比，我们的方法可以合成更高质量的新视图。 我们还展示了我们在野外图像的零样本 NeRF 合成中的普遍性。
## Nov27 - Dec3, 2022
  - [StegaNeRF：在神经辐射场中嵌入不可见信息](https://arxiv.org/abs/2212.01602) | [***``[code]``***](https://github.com/XGGNet/StegaNeRF)
    > 神经渲染的最新进展意味着通过共享 NeRF 模型权重广泛分布视觉数据的未来。 然而，虽然常见的视觉数据（图像和视频）具有明确或巧妙地嵌入所有权或版权信息的标准方法，但对于新兴的 NeRF 格式，该问题仍未得到探索。 我们介绍了 StegaNeRF，这是一种在 NeRF 渲染中嵌入隐写信息的方法。 我们设计了一个优化框架，允许从 NeRF 渲染的图像中准确提取隐藏信息，同时保留其原始视觉质量。 我们在几个潜在的部署场景下对我们的方法进行了实验评估，并进一步讨论了通过我们的分析发现的见解。 StegaNeRF 标志着对将可定制、不可察觉和可恢复的信息灌输到 NeRF 渲染的新问题的初步探索，同时对渲染图像的影响最小。 项目页面：此 https 网址。
  - [LatentSwap3D：3D 图像 GAN 的语义编辑](https://arxiv.org/abs/2212.01381) | [***``[code]``***](https://github.com/enisimsar/latentswap3d)
    > 最近的 3D 感知 GAN 依靠体积渲染技术来解开物体的姿势和外观，事实上生成整个 3D 体积而不是从潜在代码生成单视图 2D 图像。 复杂的图像编辑任务可以在基于标准 2D 的 GAN（例如，StyleGAN 模型）中作为对潜在维度的操作来执行。 然而，据我们所知，对于 3D 感知 GAN 模型，仅部分探索了类似的属性。 这项工作旨在通过展示现有方法的局限性并提出 LatentSwap3D 来填补这一空白，LatentSwap3D 是一种与模型无关的方法，旨在在预训练的 3D 感知 GAN 的潜在空间中启用属性编辑。 我们首先根据随机森林分类器的特征重要性排名，确定控制目标属性的模型的潜在空间中最相关的维度。 然后，为了应用转换，我们将正在编辑的图像的前 K 个最相关的潜在维度与显示所需属性的图像交换。 尽管它很简单，但 LatentSwap3D 以一种分离的方式提供了卓越的语义编辑，并且在质量和数量上都优于其他方法。 我们在各种 3D 感知生成模型（如 pi-GAN、GIRAFFE、StyleSDF、MVCGAN、EG3D 和 VolumeGAN）以及各种数据集（如 FFHQ、AFHQ、Cats、MetFaces 和 CompCars）上展示了我们的语义编辑方法。 可以找到项目页面：\url{this https URL}。
  - [DiffRF：渲染引导的 3D 辐射场扩散](https://arxiv.org/abs/2212.01206) | [code]
    > 我们介绍了 DiffRF，这是一种基于去噪扩散概率模型的 3D 辐射场合成新方法。 虽然现有的基于扩散的方法对图像、潜在代码或点云数据进行操作，但我们是第一个直接生成体积辐射场的方法。 为此，我们提出了一种直接在显式体素网格表示上运行的 3D 去噪模型。 然而，由于从一组姿势图像生成的辐射场可能不明确且包含伪影，因此获取地面真实辐射场样本并非易事。 我们通过将去噪公式与渲染损失配对来解决这一挑战，使我们的模型能够学习有利于良好图像质量的偏差先验，而不是试图复制像浮动伪影这样的拟合错误。 与 2D 扩散模型相比，我们的模型学习多视图一致先验，支持自由视图合成和准确的形状生成。 与 3D GAN 相比，我们基于扩散的方法自然可以在推理时启用条件生成，例如掩蔽完成或单视图 3D 合成。
  - [SparseFusion：蒸馏 View-conditioned Diffusion 用于 3D 重建](https://arxiv.org/abs/2212.00792) | [code]
    > 我们提出了 SparseFusion，这是一种稀疏视图 3D 重建方法，它统一了神经渲染和概率图像生成方面的最新进展。 现有方法通常建立在具有重新投影特征的神经渲染上，但无法生成看不见的区域或处理大视点变化下的不确定性。 替代方法将其视为（概率）2D 合成任务，虽然它们可以生成似是而非的 2D 图像，但它们无法推断出一致的底层 3D。 然而，我们发现 3D 一致性和概率图像生成之间的这种权衡并不需要存在。 事实上，我们表明几何一致性和生成推理可以在模式搜索行为中互补。 通过从视图条件潜在扩散模型中提取 3D 一致场景表示，我们能够恢复一个合理的 3D 表示，其渲染既准确又逼真。 我们评估了 CO3D 数据集中 51 个类别的方法，并表明它在失真和感知指标方面优于现有方法，用于稀疏视图新视图合成。
  - [Score Jacobian Chaining：为 3D 生成提升预训练的 2D 扩散模型](https://arxiv.org/abs/2212.00774) | [code]
    > 扩散模型学习预测梯度矢量场。 我们建议对学习到的梯度应用链式法则，并通过可微分渲染器的雅可比矩阵反向传播扩散模型的分数，我们将其实例化为体素辐射场。 此设置将多个摄像机视点的 2D 分数聚合为 3D 分数，并将预训练的 2D 模型重新用于 3D 数据生成。 我们确定了此应用程序中出现的分布不匹配的技术挑战，并提出了一种新颖的估计机制来解决它。 我们在几个现成的扩散图像生成模型上运行我们的算法，包括最近发布的在大规模 LAION 数据集上训练的稳定扩散。
  - [3D-LDM：使用潜在扩散模型生成神经隐式 3D 形状](https://arxiv.org/abs/2212.00842) | [code]
    > 扩散模型在图像生成方面显示出巨大的潜力，在生成多样性方面击败了 GAN，具有可比的图像质量。 然而，它们在 3D 形状上的应用仅限于点或体素表示，这些表示在实践中不能准确地表示 3D 表面。 我们提出了一种用于在自动解码器的潜在空间中运行的 3D 形状的神经隐式表示的扩散模型。 这使我们能够生成多样化和高质量的 3D 表面。 我们还表明，我们可以根据图像或文本调节我们的模型，以使用 CLIP 嵌入实现图像到 3D 生成和文本到 3D 生成。 此外，将噪声添加到现有形状的潜在代码中可以让我们探索形状变化。
  - [用于交互式自由视点视频的高效神经辐射场, SIGGRAPH-Asia2022](https://dl.acm.org/doi/abs/10.1145/3550469.3555376) | [code]
    > 本文旨在解决高效制作交互式自由视点视频的挑战。 最近的一些工作为神经辐射场配备了图像编码器，使它们能够跨场景进行泛化。 在处理动态场景时，他们可以简单地将每个视频帧视为一个单独的场景，并进行新颖的视图合成以生成自由视点视频。 但是，它们的渲染过程很慢，不能支持交互式应用程序。 一个主要因素是他们在推断辐射场时在空白空间中采样大量点。 我们提出了一种称为 ENeRF 的新颖场景表示，用于快速创建交互式自由视点视频。 具体来说，给定一帧的多视图图像，我们首先构建级联成本量来预测场景的粗略几何形状。 粗糙的几何体允许我们在场景表面附近采样几个点，从而显着提高渲染速度。 这个过程是完全可微的，使我们能够从 RGB 图像中共同学习深度预测和辐射场网络。 对多个基准的实验表明，我们的方法表现出有竞争力的性能，同时比以前的可推广辐射场方法至少快 60 倍。
  - [一种轻松教授变形金刚多视图几何的方法](https://arxiv.org/abs/2211.15107) | [code]
    > 变形金刚是强大的视觉学习者，这在很大程度上是因为它们明显缺乏手动指定的先验。 由于 3D 形状和视点的近乎无限可能的变化（需要灵活性）以及射影几何的精确性质（遵守刚性法则），这种灵活性在涉及多视图几何的任务中可能会出现问题。 为了解决这个难题，我们提出了一种“轻触”方法，引导视觉变形金刚学习多视图几何，但允许它们在需要时摆脱束缚。 我们通过使用极线来引导 Transformer 的交叉注意力图来实现这一点，惩罚极线外的注意力值并鼓励沿着这些线的更高注意力，因为它们包含几何上合理的匹配。 与以前的方法不同，我们的建议在测试时不需要任何相机姿势信息。 我们专注于姿势不变的对象实例检索，由于查询和检索图像之间的视点存在巨大差异，因此标准 Transformer 网络在这方面存在困难。 在实验上，我们的方法在对象检索方面优于最先进的方法，而且在测试时不需要姿势信息。
  - [通过伪多视图优化的高保真 3D GAN 反演](https://arxiv.org/abs/2211.15662) | [***``[code]``***](https://github.com/jiaxinxie97/HFGI3D)
    > 我们提出了一个高保真 3D 生成对抗网络 (GAN) 反演框架，可以在保留输入图像的特定细节的同时合成逼真的新视图。 由于高保真 3D 反演中的几何纹理权衡，高保真 3D GAN 反演本质上具有挑战性，其中对单个视图输入图像的过度拟合通常会在潜在优化期间损坏估计的几何形状。 为了解决这一挑战，我们提出了一种新的管道，它建立在具有可见性分析的伪多视图估计之上。 我们保留可见部分的原始纹理，并对被遮挡的部分使用生成先验。 广泛的实验表明，我们的方法比最先进的方法实现了有利的重建和新颖的视图合成质量，即使对于具有分布外纹理的图像也是如此。 拟议的管道还支持使用反向潜代码和 3D 感知纹理修改进行图像属性编辑。 我们的方法可以从单个图像进行高保真 3D 渲染，这有望用于 AI 生成的 3D 内容的各种应用。
## Nov20 - Nov26, 2022
  - [通过神经渲染的无监督连续语义适应](https://arxiv.org/abs/2211.13969) | [code]
    > 越来越多的应用程序依赖于数据驱动模型，这些模型被部署用于跨一系列场景的感知任务。由于训练和部署数据之间的不匹配，在新场景上调整模型对于获得良好性能通常至关重要。在这项工作中，我们研究了语义分割任务的持续多场景适应，假设在部署期间没有可用的地面实况标签，并且应该保持先前场景的性能。我们建议通过融合分割模型的预测，然后使用视图一致的渲染语义标签作为伪标签来调整模型，为每个场景训练一个语义 NeRF 网络。通过与分割模型的联合训练，Semantic-NeRF 模型有效地实现了 2D-3D 知识迁移。此外，由于其紧凑的尺寸，它可以存储在长期记忆中，随后用于从任意角度渲染数据以减少遗忘。我们在 ScanNet 上评估了我们的方法，我们的方法优于基于体素的基线和最先进的无监督域适应方法。
  - [ShadowNeuS：Shadow Ray 监督的神经 SDF 重建](https://arxiv.org/abs/2211.14086) | [code]
    > 通过监督场景和多视图图像平面之间的相机光线，NeRF 为新视图合成任务重建神经场景表示。另一方面，光源和场景之间的阴影光线还有待考虑。因此，我们提出了一种新颖的阴影射线监督方案，可以优化沿射线的样本和射线位置。通过监督阴影光线，我们在多种光照条件下成功地从单视图纯阴影或 RGB 图像重建场景的神经 SDF。给定单视图二进制阴影，我们训练神经网络重建不受相机视线限制的完整场景。通过进一步模拟图像颜色和阴影光线之间的相关性，我们的技术还可以有效地扩展到 RGB 输入。我们将我们的方法与之前关于从单视图二值阴影或 RGB 图像重建形状的挑战性任务的工作进行比较，并观察到显着的改进。代码和数据将被发布。
  - [Peekaboo：文本到图像扩散模型是零样本分割器](https://arxiv.org/abs/2211.13224) | [code]
    > 最近基于扩散的生成模型与视觉语言模型相结合，能够根据自然语言提示创建逼真的图像。虽然这些模型是在大型互联网规模的数据集上训练的，但这种预训练模型并没有直接引入任何语义定位或基础。大多数当前的定位或接地方法都依赖于边界框或分割掩码形式的人工注释定位信息。例外是一些无监督方法，它们利用面向本地化的体系结构或损失函数，但它们需要单独训练。在这项工作中，我们探索了现成的扩散模型，在没有接触此类定位信息的情况下进行训练，如何能够在没有特定于分段的重新训练的情况下建立各种语义短语。引入了推理时间优化过程，能够生成以自然语言为条件的分割掩码。我们评估了我们在 Pascal VOC 数据集上进行无监督语义分割的提案 Peekaboo。此外，我们评估了 RefCOCO 数据集上的引用分割。总之，我们提出了第一个零样本、开放词汇、无监督（无定位信息）、语义基础技术，利用基于扩散的生成模型，无需重新训练。我们的代码将公开发布。
  - [PANeRF：基于少样本输入的改进神经辐射场的伪视图增强](https://arxiv.org/abs/2211.12758) | [code]
    > 近年来开发了神经辐射场 (NeRF) 方法，该技术在合成复杂场景的新视图方面具有广阔的应用前景。然而，NeRF 需要密集的输入视图，通常有数百个，以生成高质量图像。随着输入视图数量的减少，NeRF 对未见视点的渲染质量趋于急剧下降。为了克服这一挑战，我们提出了 NeRF 的伪视图增强，该方案通过考虑少镜头输入的几何形状来扩展足够数量的数据。我们首先通过利用扩展的伪视图来初始化 NeRF 网络，这可以有效地减少渲染看不见的视图时的不确定性。随后，我们通过使用包含精确几何和颜色信息的稀疏视图输入来微调网络。通过各种设置下的实验，我们验证了我们的模型忠实地合成了高质量的新视图图像，并且优于现有的多视图数据集方法。
  - [零 NeRF：零重叠注册](https://arxiv.org/abs/2211.12544) | [code]
    > 我们提出了零 NeRF，这是一种投影表面配准方法，据我们所知，它提供了第一个能够在具有最小或零视觉对应的场景表示之间对齐的通用解决方案。为此，我们加强了部分和完整重建的可见表面之间的一致性，这使我们能够约束被遮挡的几何体。我们使用 NeRF 作为我们的表面表示和 NeRF 渲染管道来执行此对齐。为了证明我们方法的有效性，我们从对面的现实世界场景中注册了无法使用现有方法准确注册的无限小重叠，并将这些结果与广泛使用的注册方法进行了比较。
  - [SPARF：来自稀疏和嘈杂姿势的神经辐射场](https://arxiv.org/abs/2211.11738) | [code]
    > 神经辐射场 (NeRF) 最近已成为合成逼真新颖视图的有力代表。虽然表现出令人印象深刻的性能，但它依赖于具有高精度相机姿势的密集输入视图的可用性，从而限制了其在现实场景中的应用。在这项工作中，我们引入了稀疏姿态调整辐射场 (SPARF)，以应对仅在少量宽基线输入图像（低至 3 张）且相机姿态嘈杂的情况下进行新视图合成的挑战。我们的方法利用多视图几何约束来共同学习 NeRF 并改进相机姿势。通过依赖于输入视图之间提取的像素匹配，我们的多视图对应目标强制优化场景和相机姿势以收敛到全局和几何精确的解决方案。我们的深度一致性损失进一步鼓励重建的场景从任何角度来看都是一致的。我们的方法在多个具有挑战性的数据集的稀疏视图机制中设置了一个新的技术状态。
## Nov13 - Nov19, 2022
  - [Magic3D：高分辨率文本到 3D 内容创建](https://arxiv.org/abs/2211.10440) | [code]
    > DreamFusion 最近展示了预训练的文本到图像扩散模型在优化神经辐射场 (NeRF) 方面的实用性，实现了卓越的文本到 3D 合成结果。然而，该方法有两个固有的局限性：(a) NeRF 的优化极其缓慢和 (b) NeRF 上的低分辨率图像空间监督，导致处理时间长的低质量 3D 模型。在本文中，我们通过使用两阶段优化框架来解决这些限制。首先，我们使用低分辨率扩散先验获得粗糙模型，并使用稀疏 3D 哈希网格结构进行加速。使用粗略表示作为初始化，我们进一步优化了带纹理的 3D 网格模型，该模型具有与高分辨率潜在扩散模型交互的高效可微分渲染器。我们的方法被称为 Magic3D，可以在 40 分钟内创建高质量的 3D 网格模型，比 DreamFusion 快 2 倍（据报道平均需要 1.5 小时），同时还实现了更高的分辨率。用户研究表明 61.7% 的评分者更喜欢我们的方法而不是 DreamFusion。连同图像调节生成功能，我们为用户提供了控制 3D 合成的新方法，为各种创意应用开辟了新途径。
  - [RenderDiffusion：用于 3D 重建、修复和生成的图像扩散](https://arxiv.org/abs/2211.09869) | [code]
    > 扩散模型目前在条件和无条件图像生成方面都达到了最先进的性能。然而，到目前为止，图像扩散模型不支持 3D 理解所需的任务，例如视图一致的 3D 生成或单视图对象重建。在本文中，我们将 RenderDiffusion 作为第一个用于 3D 生成和推理的扩散模型，可以仅使用单眼 2D 监督进行训练。我们方法的核心是一种新颖的图像去噪架构，它在每个去噪步骤中生成并渲染场景的中间三维表示。这在扩散过程中强制实施了一个强大的归纳结构，为我们提供了一个 3D 一致的表示，同时只需要 2D 监督。可以从任何视点渲染生成的 3D 表示。我们在 ShapeNet 和 Clevr 数据集上评估 RenderDiffusion，并展示了在生成 3D 场景和从 2D 图像推断 3D 场景方面的竞争性能。此外，我们基于扩散的方法允许我们使用 2D 修复来编辑 3D 场景。我们相信，我们的工作有望在对大量图像集进行训练时实现大规模的完整 3D 生成，从而避免对大型 3D 模型集进行监督的需要。
## Nov6 - Nov12, 2022
  - [3D常见宠物：现实生活中可变形类别的动态新视角合成](https://arxiv.org/abs/2211.03889) | [code]
    > 从稀疏视图中获得对象的逼真重建本质上是模棱两可的，只能通过学习合适的重建先验来实现。早期关于稀疏刚性对象重建的工作成功地从大型数据集（如 CO3D）中学习了这样的先验。在本文中，我们将这种方法扩展到动态对象。我们以猫和狗作为代表性示例，并介绍 Common Pets in 3D (CoP3D)，这是一组众包视频，展示了大约 4,200 种不同的宠物。 CoP3D 是首批用于“野外”非刚性 3D 重建基准测试的大型数据集之一。我们还提出了 Tracker-NeRF，这是一种从我们的数据集中学习 4D 重建的方法。在测试时，给定一个看不见的物体的少量视频帧，Tracker-NeRF 预测其 3D 点的轨迹并生成新视图、插值视点和时间。 CoP3D 的结果揭示了比现有基线更好的非刚性新视图合成性能。
## Oct30 - Nov5, 2022
  - [用于机器人操纵的神经抓取距离场](https://arxiv.org/abs/2211.02647) | [code]
    > 我们将抓取学习制定为一个神经场，并提出神经抓取距离场 (NGDF)。这里，输入是机器人末端执行器的 6D 姿态，输出是到物体有效抓握的连续流形的距离。与预测一组离散候选抓握的当前方法相比，基于距离的 NGDF 表示很容易被解释为成本，并且最小化该成本会产生成功的抓握姿势。这种抓取距离成本可以直接合并到轨迹优化器中，与其他成本（如轨迹平滑度和碰撞避免）进行联合优化。在优化过程中，随着各种成本的平衡和最小化，抓取目标可以平滑变化，因为学习到的抓取域是连续的。在使用 Franka 手臂的模拟基准测试中，我们发现使用 NGDF 的联合抓取和规划比基线执行成功率高出 63%，同时泛化到看不见的查询姿势和看不见的物体形状。项目页面：此 https 网址。
## Oct23 - Oct29, 2022
  - [Compressing Explicit Voxel Grid Representations：快速的 NeRFs 也变小了](https://arxiv.org/abs/2210.12782) | [code]
    > 由于其固有的紧凑性，NeRF 彻底改变了逐场景辐射场重建的世界。 NeRF 的主要限制之一是它们在训练和推理时的渲染速度都很慢。最近的研究重点是优化表示场景的显式体素网格 (EVG)，它可以与神经网络配对以学习辐射场。这种方法显着提高了训练和推理时间的速度，但代价是占用大量内存。在这项工作中，我们提出了 Re:NeRF，这是一种专门针对 EVG-NeRF 可压缩性的方法，旨在减少 NeRF 模型的内存存储，同时保持相当的性能。我们在四种流行的基准测试中使用三种不同的 EVG-NeRF 架构对我们的方法进行了基准测试，展示了 Re:NeRF 广泛的可用性和有效性。
## Oct16 - Oct22, 2022
  - [TANGO：通过光照分解实现文本驱动的真实感和强大的 3D 风格化, NeurIPS2022](https://arxiv.org/abs/2210.11277) | [***``[code]``***](https://cyw-3d.github.io/tango/)
    > 通过程式化创建 3D 内容是计算机视觉和图形研究中一个有前途但具有挑战性的问题。在这项工作中，我们专注于对任意拓扑的给定表面网格的逼真外观渲染进行风格化。受最近对比语言-图像预训练 (CLIP) 模型的跨模态监督激增的启发，我们提出了 TANGO，它根据文本提示以逼真的方式转移给定 3D 形状的外观风格。从技术上讲，我们建议将外观风格分解为空间变化的双向反射率分布函数、局部几何变化和照明条件，通过基于球形高斯的可微分渲染器通过监督 CLIP 损失来共同优化它们。因此，TANGO 通过自动预测​​反射效果来实现逼真的 3D 风格转换，即使是对于裸露的、低质量的网格，也无需对特定任务的数据集进行培训。大量实验表明，TANGO 在逼真的质量、3D 几何的一致性和对低质量网格进行样式化时的鲁棒性方面优于现有的文本驱动 3D 样式转换方法。我们的代码和结果可在我们的项目网页 https URL 上找到。
  - [坐标并不孤单——码本先验有助于隐式神经 3D 表示, NeurIPS2022](https://arxiv.org/abs/2210.11170) | [code]
    > 隐式神经 3D 表示在表面或场景重建和新颖的视图合成中取得了令人印象深刻的结果，这通常使用基于坐标的多层感知器 (MLP) 来学习连续的场景表示。然而，现有的方法，例如神经辐射场 (NeRF) 及其变体，通常需要密集的输入视图（即 50-150）才能获得不错的结果。为了重温对大量校准图像的过度依赖并丰富基于坐标的特征表示，我们探索将先验信息注入基于坐标的网络，并引入一种新颖的基于坐标的模型 CoCo-INR，用于隐式神经 3D 表示。我们方法的核心是两个注意力模块：码本注意力和坐标注意力。前者从先验码本中提取包含丰富几何和外观信息的有用原型，后者将这些先验信息传播到每个坐标中，并丰富其对场景或物体表面的特征表示。在先验信息的帮助下，与使用较少可用校准图像的当前方法相比，我们的方法可以渲染具有更逼真外观和几何形状的 3D 视图。在包括 DTU 和 BlendedMVS 在内的各种场景重建数据集以及完整的 3D 头部重建数据集 H3DS 上的实验证明了我们提出的方法在较少输入视图下的鲁棒性和精细的细节保留能力。
## Oct9 - Oct15, 2022
  - [AniFaceGAN：用于视频头像的动画 3D 感知人脸图像生成, NeurIPS2022](https://arxiv.org/abs/2210.06465) | [***``[code]``***](https://yuewuhkust.github.io/AniFaceGAN/files/github_icon.jpeg)
    > 尽管 2D 生成模型在人脸图像生成和动画方面取得了长足进步，但它们在从不同相机视点渲染图像时经常会遇到不希望的伪影，例如 3D 不一致。这可以防止他们合成与真实动画无法区分的视频动画。最近，3D 感知 GAN 扩展了 2D GAN，通过利用 3D 场景表示来明确解开相机姿势。这些方法可以很好地保持生成图像在不同视图中的 3D 一致性，但它们无法实现对其他属性的细粒度控制，其中面部表情控制可以说是面部动画最有用和最理想的方法。在本文中，我们提出了一种可动画的 3D 感知 GAN，用于多视图一致的人脸动画生成。关键思想是将 3D-aware GAN 的 3D 表示分解为模板字段和变形字段，其中前者用规范表达式表示不同的身份，后者表征每个身份的表达变化。为了通过变形实现对面部表情的有意义的控制，我们在 3D 感知 GAN 的对抗训练期间提出了生成器和参数 3D 面部模型之间的 3D 级模仿学习方案。这有助于我们的方法实现具有强烈视觉 3D 一致性的高质量动画人脸图像生成，即使仅使用非结构化 2D 图像进行训练。广泛的实验证明了我们优于以前的工作的性能。项目页面：此 https 网址
  - [LION：用于 3D 形状生成的潜在点扩散模型, NeurIPS2022](https://arxiv.org/abs/2210.06978) | [***``[code]``***](https://nv-tlabs.github.io/LION)
    > 去噪扩散模型 (DDM) 在 3D 点云合成中显示出可喜的结果。为了推进 3D DDM 并使它们对数字艺术家有用，我们需要 (i) 高生成质量，(ii) 操作和应用的灵活性，例如条件合成和形状插值，以及 (iii) 输出平滑表面或网格的能力。为此，我们介绍了用于 3D 形状生成的分层潜在点扩散模型 (LION)。 LION 被设置为具有分层潜在空间的变分自动编码器 (VAE)，该分层潜在空间将全局形状潜在表示与点结构潜在空间相结合。对于生成，我们在这些潜在空间中训练两个分层 DDM。与直接在点云上运行的 DDM 相比，分层 VAE 方法提高了性能，而点结构的潜在模型仍然非常适合基于 DDM 的建模。在实验上，LION 在多个 ShapeNet 基准上实现了最先进的生成性能。此外，我们的 VAE 框架使我们能够轻松地将 LION 用于不同的相关任务：LION 在多模态形状去噪和体素条件合成方面表现出色，并且可以适应文本和图像驱动的 3D 生成。我们还演示了形状自动编码和潜在形状插值，并使用现代表面重建技术增强了 LION 以生成平滑的 3D 网格。我们希望 LION 凭借其高质量的生成、灵活性和表面重建功能，为处理 3D 形状的艺术家提供强大的工具。项目页面和代码：此 https 网址。
  - [CLIP-Fields：机器人记忆的弱监督语义场](https://mahis.life/clip-fields/) | [code]
    > 我们提出了 CLIP-Fields，这是一种隐式场景模型，可以在没有直接人工监督的情况下进行训练。该模型学习从空间位置到语义嵌入向量的映射。然后，该映射可用于各种任务，例如分割、实例识别、空间语义搜索和视图定位。最重要的是，映射可以通过仅来自网络图像和网络文本训练模型（如 CLIP、Detic 和 Sentence-BERT）的监督进行训练。与 Mask-RCNN 之类的基线相比，我们的方法在 HM3D 数据集上的少量实例识别或语义分割方面表现优于仅一小部分示例。最后，我们展示了使用 CLIP-Fields 作为场景记忆，机器人可以在现实环境中执行语义导航。我们的代码和演示可在此处获得：https://mahis.life/clip-fields/
## Oct2 - Oct8, 2022
  - [用于新视图合成的自我改进多平面到层图像, WACV2023](https://samsunglabs.github.io/MLI/) | [***``[code]``***](https://github.com/SamsungLabs/MLI)
    > 我们提出了一种用于轻量级小说视图合成的新方法，该方法可以推广到任意前向场景。最近的方法在计算上很昂贵，需要逐场景优化，或者产生内存昂贵的表示。我们首先用一组正面平行的半透明平面来表示场景，然后以端到端的方式将它们转换为可变形层。此外，我们采用前馈细化程序，通过聚合来自输入视图的信息来纠正估计的表示。我们的方法在处理新场景时不需要微调，并且可以不受限制地处理任意数量的视图。实验结果表明，我们的方法在常用指标和人工评估方面超过了最近的模型，在推理速度和推断分层几何的紧凑性方面具有显着优势，请参阅此 https URL
  - [用于隐式场景重建的不确定性驱动的主动视觉](https://arxiv.org/abs/2210.00978) | [code]
    > 多视图隐式场景重建方法由于能够表示复杂的场景细节而变得越来越流行。最近的努力致力于改进输入信息的表示并减少获得高质量重建所需的视图数量。然而，也许令人惊讶的是，关于选择哪些视图以最大限度地提高场景理解的研究在很大程度上仍未得到探索。我们提出了一种用于隐式场景重建的不确定性驱动的主动视觉方法，该方法利用体积渲染在场景中累积的占用不确定性来选择下一个要获取的视图。为此，我们开发了一种基于占用的重建方法，该方法使用 2D 或 3D 监督准确地表示场景。我们在 ABC 数据集和野外 CO3D 数据集上评估了我们提出的方法，并表明：（1）我们能够获得高质量的最先进的占用重建； (2) 我们的视角条件不确定性定义有效地推动了下一个最佳视图选择的改进，并且优于强大的基线方法； (3) 我们可以通过对视图选择候选执行基于梯度的搜索来进一步提高形状理解。总体而言，我们的结果突出了视图选择对于隐式场景重建的重要性，使其成为进一步探索的有希望的途径。
  - [SinGRAV：从单个自然场景中学习生成辐射量](https://arxiv.org/abs/2210.01202) | [code]
    > 我们提出了一个用于一般自然场景的 3D 生成模型。由于缺乏表征目标场景的必要 3D 数据量，我们建议从单个场景中学习。我们的关键见解是，一个自然场景通常包含多个组成部分，其几何、纹理和空间排列遵循一些清晰的模式，但在同一场景中的不同区域仍然表现出丰富的变化。这表明将生成模型的学习本地化在大量局部区域上。因此，我们利用具有空间局部性偏差的多尺度卷积网络来学习单个场景中多个尺度的局部区域的统计信息。与现有方法相比，我们的学习设置绕过了从许多同质 3D 场景中收集数据以学习共同特征的需要。我们创造了我们的方法 SinGRAV，用于从单个自然场景中学习生成辐射体积。我们展示了 SinGRAV 从单个场景生成合理多样的变化的能力，SingGRAV 相对于最先进的生成神经场景方法的优点，以及 SinGRAV 在各种应用中的多功能性，涵盖 3D 场景编辑、合成和动画。代码和数据将被发布以促进进一步的研究。
  - [IntrinsicNeRF：学习用于可编辑新视图合成的内在神经辐射场](https://arxiv.org/abs/2210.00647) | [***``[code]``***](https://github.com/zju3dv/IntrinsicNeRF)
    > 我们提出了被称为 IntrinsicNeRF 的内在神经辐射场，它将内在分解引入到基于 NeRF 的~\cite{mildenhall2020nerf} 神经渲染方法中，并且可以在现有的逆向渲染结合神经渲染方法的同时在房间规模的场景中执行可编辑的新视图合成~ \cite{zhang2021physg, zhang2022modeling} 只能用于特定对象的场景。鉴于内在分解本质上是一个模棱两可且约束不足的逆问题，我们提出了一种新颖的距离感知点采样和自适应反射率迭代聚类优化方法，该方法使具有传统内在分解约束的 IntrinsicNeRF 能够以无监督的方式进行训练，从而在时间上一致的内在分解结果。为了解决场景中相似反射率的不同相邻实例被错误地聚集在一起的问题，我们进一步提出了一种从粗到细优化的层次聚类方法，以获得快速的层次索引表示。它支持引人注目的实时增强现实应用，例如场景重新着色、材质编辑和照明变化。 Blender 对象和副本场景的大量实验表明，即使对于具有挑战性的序列，我们也可以获得高质量、一致的内在分解结果和高保真新视图合成。项目网页上提供了代码和数据：此 https 网址。
## Sep25 - Oct1, 2022
  - [通过对极约束不带姿势相机的结构感知 NeRF](https://arxiv.org/abs/2210.00183) | [***``[code]``***](https://github.com/XTU-PR-LAB/SaNerf)
    > 用于逼真的新视图合成的神经辐射场 (NeRF) 需要通过运动结构 (SfM) 方法预先获取相机姿势。这种两阶段策略使用不方便并且会降低性能，因为姿势提取中的错误会传播到视图合成。我们将姿势提取和视图合成集成到一个端到端的过程中，这样它们就可以相互受益。为了训练 NeRF 模型，只给出了 RGB 图像，没有预先知道的相机姿势。相机位姿是通过极线约束获得的，其中不同视图中的相同特征具有根据提取的位姿从本地相机坐标转换而来的相同世界坐标。对极约束与像素颜色约束联合优化。姿势由基于 CNN 的深度网络表示，其输入是相关帧。这种联合优化使 NeRF 能够感知场景的结构，从而提高泛化性能。在各种场景上进行的大量实验证明了所提出方法的有效性。此 https 网址提供了代码。
  - [使用几何感知鉴别器改进 3D 感知图像合成, NeurIPS2022](https://arxiv.org/abs/2209.15637) | [***``[code]``***](https://github.com/vivianszf/geod)
    > 3D 感知图像合成旨在学习一个生成模型，该模型可以渲染逼真的 2D 图像，同时捕捉体面的底层 3D 形状。一种流行的解决方案是采用生成对抗网络 (GAN)，并用 3D 渲染器替换生成器，其中通常使用带有神经辐射场 (NeRF) 的体积渲染。尽管合成质量有所提高，但现有方法无法获得适度的 3D 形状。我们认为，考虑到 GAN 公式中的两人游戏，仅使生成器具有 3D 感知能力是不够的。换句话说，取代生成机制只能提供生成 3D 感知图像的能力，但不能保证，因为生成器的监督主要来自鉴别器。为了解决这个问题，我们提出 GeoD 通过学习几何感知鉴别器来改进 3D 感知 GAN。具体来说，除了从 2D 图像空间中区分真假样本外，还要求鉴别器从输入中获取几何信息，然后将其用作生成器的指导。这种简单而有效的设计有助于学习更准确的 3D 形状。对各种生成器架构和训练数据集的广泛实验验证了 GeoD 优于最先进的替代方案。此外，我们的方法被注册为一个通用框架，这样一个更有能力的鉴别器（即，除了域分类和几何提取之外，还有第三个新的视图合成任务）可以进一步帮助生成器获得更好的多视图一致性。
  - [MonoNeuralFusion：具有几何先验的在线单目神经 3D 重建](https://arxiv.org/abs/2209.15153) | [code]
    > 从单目视频重建高保真 3D 场景仍然具有挑战性，特别是对于完整和细粒度的几何重建。先前具有神经隐式表示的 3D 重建方法已显示出完整场景重建的有希望的能力，但它们的结果通常过于平滑且缺乏足够的几何细节。本文介绍了一种新颖的神经隐式场景表示法，用于从单目视频中进行高保真在线 3D 场景重建的体积渲染。对于细粒度重建，我们的关键见解是将几何先验纳入神经隐式场景表示和神经体绘制，从而产生基于体绘制优化的有效几何学习机制。受益于此，我们提出了 MonoNeuralFusion 来从单目视频执行在线神经 3D 重建，从而在动态 3D 单目扫描期间有效地生成和优化 3D 场景几何图形。与最先进方法的广泛比较表明，我们的 MonoNeuralFusion 在数量和质量上始终生成更好的完整和细粒度的重建结果。
  - [SymmNeRF：学习探索单视图视图合成的对称先验, ACCV2022](https://arxiv.org/abs/2209.14819) | [***``[code]``***](https://github.com/xingyi-li/SymmNeRF)
    > 我们研究了从单个图像中对对象进行新视图合成的问题。现有方法已经证明了单视图视图合成的潜力。但是，它们仍然无法恢复精细的外观细节，尤其是在自闭区域。这是因为单个视图仅提供有限的信息。我们观察到人造物体通常表现出对称的外观，这会引入额外的先验知识。受此启发，我们研究了将对称性显式嵌入场景表示的潜在性能增益。在本文中，我们提出了 SymmNeRF，这是一种基于神经辐射场 (NeRF) 的框架，在引入对称先验的情况下结合了局部和全局条件。特别是，SymmNeRF 将像素对齐的图像特征和相应的对称特征作为 NeRF 的额外输入，其参数由超网络生成。由于参数以图像编码的潜在代码为条件，因此 SymmNeRF 与场景无关，可以推广到新场景。对合成数据集和真实世界数据集的实验表明，SymmNeRF 可以合成具有更多细节的新颖视图，而不管姿势变换如何，并且在应用于看不见的对象时表现出良好的泛化性。代码位于：此 https URL。
  - [360FusionNeRF：具有联合引导的全景神经辐射场](https://arxiv.org/abs/2209.14265) | [code]
    > 我们提出了一种基于神经辐射场 (NeRF) 从单个 360 度全景图像合成新视图的方法。类似设置中的先前研究依赖于多层感知的邻域插值能力来完成由遮挡引起的缺失区域，这导致其预测中的伪影。我们提出了 360FusionNeRF，这是一个半监督学习框架，我们在其中引入几何监督和语义一致性来指导渐进式训练过程。首先，将输入图像重新投影到 360 度图像，并在其他相机位置提取辅助深度图。除了 NeRF 颜色指导之外，深度监督还改进了合成视图的几何形状。此外，我们引入了语义一致性损失，鼓励对新视图进行逼真的渲染。我们使用预训练的视觉编码器（例如 CLIP）提取这些语义特征，CLIP 是一种视觉转换器，通过自然语言监督从网络挖掘出的数亿张不同的 2D 照片进行训练。实验表明，我们提出的方法可以在保留场景特征的同时产生未观察到的区域的合理完成。在跨各种场景进行训练时，360FusionNeRF 在转移到合成 Structured3D 数据集（PSNR~5%，SSIM~3% LPIPS~13%）、真实世界的 Matterport3D 数据集（PSNR~3%）时始终保持最先进的性能, SSIM~3% LPIPS~9%) 和 Replica360 数据集 (PSNR~8%, SSIM~2% LPIPS~18%)。
## Sep18 - Sep24, 2022
  - [PNeRF：用于不确定 3D 视觉映射的概率神经场景表示, ICRA2023](https://arxiv.org/abs/2209.11677) | [code]
    > 最近，神经场景表示在视觉上表示 3D 场景提供了非常令人印象深刻的结果，但是，它们的研究和进展主要局限于计算机图形中虚拟模型的可视化或计算机视觉中的场景重建，而没有明确考虑传感器和姿势的不确定性。然而，在机器人应用中使用这种新颖的场景表示需要考虑神经图中的这种不确定性。因此，本文的目的是提出一种用不确定的训练数据训练 {\em 概率神经场景表示} 的新方法，该方法可以将这些表示包含在机器人应用程序中。使用相机或深度传感器获取图像包含固有的不确定性，此外，用于学习 3D 模型的相机姿势也不完善。如果将这些测量值用于训练而不考虑其不确定性，则生成的模型不是最优的，并且生成的场景表示可能包含诸如模糊和几何不均匀等伪影。在这项工作中，通过关注以概率方式使用不确定信息进行训练，研究了将不确定性整合到学习过程中的问题。所提出的方法涉及使用不确定性项显式增加训练似然性，使得网络的学习概率分布相对于训练不确定性最小化。将会显示，除了更精确和一致的几何形状之外，这会导致更准确的图像渲染质量。已经对合成数据集和真实数据集进行了验证，表明所提出的方法优于最先进的方法。结果表明，即使在训练数据有限的情况下，所提出的方法也能够呈现新颖的高质量视图。
  - [ActiveNeRF：通过不确定性估计学习在哪里看](https://arxiv.org/abs/2209.08546) | [***``[code]``***](https://github.com/LeapLabTHU/ActiveNeRF)
    > 最近，神经辐射场 (NeRF) 在重建 3D 场景和从一组稀疏的 2D 图像合成新视图方面显示出令人鼓舞的性能。尽管有效，但 NeRF 的性能很大程度上受训练样本质量的影响。由于场景中的姿势图像有限，NeRF 无法很好地泛化到新颖的视图，并且可能会在未观察到的区域中崩溃为琐碎的解决方案。这使得 NeRF 在资源受限的情况下变得不切实际。在本文中，我们提出了一种新颖的学习框架 ActiveNeRF，旨在对输入预算受限的 3D 场景进行建模。具体来说，我们首先将不确定性估计纳入 NeRF 模型，以确保在少量观察下的稳健性，并提供对 NeRF 如何理解场景的解释。在此基础上，我们建议使用基于主动学习方案的新捕获样本来补充现有的训练集。通过评估给定新输入的不确定性减少情况，我们选择带来最多信息增益的样本。通过这种方式，可以用最少的额外资源提高新视图合成的质量。大量实验验证了我们的模型在真实场景和合成场景上的性能，尤其是在训练数据较少的情况下。代码将在 \url{this https URL} 发布。
## Sep11 - Sep17, 2022
  - [学习用于视图合成的统一 3D 点云](https://arxiv.org/abs/2209.05013) | [code]
    > 基于 3D 点云表示的视图合成方法已证明是有效的。然而，现有方法通常仅从单个源视图合成新视图，并且将它们泛化以处理多个源视图以追求更高的重建质量并非易事。在本文中，我们提出了一种新的基于深度学习的视图合成范式，它从不同的源视图中学习统一的 3D 点云。具体来说，我们首先通过根据深度图将源视图投影到 3D 空间来构建子点云。然后，我们通过自适应融合子点云联合上定义的局部邻域中的点来学习统一的 3D 点云。此外，我们还提出了一个 3D 几何引导图像恢复模块来填充孔洞并恢复渲染新视图的高频细节。三个基准数据集的实验结果表明，我们的方法在数量上和视觉上都在很大程度上优于最先进的视图合成方法。
## Sep4 - Sep10, 2022
## Aug28 - Sep3, 2022
  - [Dual-Space NeRF：在不同空间中学习动画化身和场景照明, 3DV2022](https://arxiv.org/abs/2208.14851) | [code]
    > 在规范空间中对人体进行建模是捕捉和动画的常见做法。但是当涉及到神经辐射场 (NeRF) 时，仅仅在标准空间中学习一个静态的 NeRF 是不够的，因为即使场景照明是恒定的，当人移动时身体的照明也会发生变化。以前的方法通过学习每帧嵌入来缓解光照的不一致性，但这种操作并不能推广到看不见的姿势。鉴于光照条件在世界空间中是静态的，而人体在规范空间中是一致的，我们提出了一种双空间 NeRF，它在两个独立的空间中使用两个 MLP 对场景光照和人体进行建模。为了弥合这两个空间，以前的方法主要依赖于线性混合蒙皮 (LBS) 算法。然而，动态神经领域的 LBS 的混合权重是难以处理的，因此通常用另一个 MLP 来记忆，这不能推广到新的姿势。尽管可以借用 SMPL 等参数网格的混合权重，但插值操作会引入更多伪影。在本文中，我们建议使用重心映射，它可以直接泛化到看不见的姿势，并且出人意料地取得了比具有神经混合权重的 LBS 更好的结果。 Human3.6M 和 ZJU-MoCap 数据集的定量和定性结果显示了我们方法的有效性。
## Aug21 - Aug27, 2022
  - [DreamBooth：为主题驱动生成微调文本到图像的扩散模型](https://dreambooth.github.io/) | [code]
    > 大型文本到图像模型在人工智能的演进中实现了显着的飞跃，能够从给定的文本提示中对图像进行高质量和多样化的合成。然而，这些模型缺乏模仿给定参考集中对象的外观并在不同上下文中合成它们的新颖再现的能力。在这项工作中，我们提出了一种“个性化”文本到图像扩散模型的新方法（专门针对用户的需求）。给定主题的几张图像作为输入，我们微调预训练的文本到图像模型（Imagen，尽管我们的方法不限于特定模型），以便它学会将唯一标识符与该特定主题绑定.一旦对象被嵌入模型的输出域中，唯一标识符就可以用于合成在不同场景中情境化的对象的完全新颖的真实感图像。通过利用嵌入在模型中的语义先验和新的自生类特定先验保存损失，我们的技术能够在参考图像中没有出现的不同场景、姿势、视图和照明条件下合成主体。我们将我们的技术应用于几个以前无懈可击的任务，包括主题重新上下文化、文本引导视图合成、外观修改和艺术渲染（同时保留主题的关键特征）。项目页面：此 https 网址
  - [E-NeRF：来自移动事件相机的神经辐射场](https://arxiv.org/abs/2208.11300) | [code]
    > 从理想图像估计神经辐射场 (NeRFs) 已在计算机视觉领域得到广泛研究。大多数方法假设最佳照明和缓慢的相机运动。这些假设在机器人应用中经常被违反，其中图像包含运动模糊并且场景可能没有合适的照明。这可能会导致下游任务（例如场景的导航、检查或可视化）出现重大问题。为了缓解这些问题，我们提出了 E-NeRF，这是第一种从快速移动的事件摄像机中以 NeRF 形式估计体积场景表示的方法。我们的方法可以在非常快速的运动和高动态范围条件下恢复 NeRF，在这种情况下，基于帧的方法会失败。我们展示了仅通过提供事件流作为输入来渲染高质量帧是可能的。此外，通过结合事件和帧，我们可以估计在严重运动模糊下比最先进的方法质量更高的 NeRF。我们还表明，在只有很少的输入视图可用的情况下，结合事件和帧可以克服 NeRF 估计的失败情况，而无需额外的正则化。
  - [FurryGAN：高质量的前景感知图像合成, ECCV2022](https://jeongminb.github.io/FurryGAN/) | [***``[code]``***](https://jeongminb.github.io/FurryGAN/)
    > 前景感知图像合成旨在生成图像及其前景蒙版。一种常见的方法是将图像公式化为前景图像和背景图像的蒙版混合。这是一个具有挑战性的问题，因为它很容易达到一个简单的解决方案，即任一图像压倒另一个图像，即蒙版完全满或空，前景和背景没有有意义地分离。我们展示了 FurryGAN 的三个关键组件：1）将前景图像和合成图像都强加为逼真，2）将掩码设计为粗略和精细掩码的组合，以及 3）通过辅助掩码预测器引导生成器鉴别器。我们的方法使用非常详细的 alpha 蒙版生成逼真的图像，这些蒙版以完全无人监督的方式覆盖头发、毛皮和胡须。
## Aug14 - Aug20, 2022
  - [通过隐式视觉引导和超网络生成文本到图像](https://arxiv.org/abs/2208.08493) | [code]
    > 我们开发了一种文本到图像生成的方法，该方法包含额外的检索图像，由隐式视觉引导损失和生成目标的组合驱动。与大多数现有的仅以文本为输入的文本到图像生成方法不同，我们的方法将跨模态搜索结果动态地馈送到统一的训练阶段，从而提高了生成结果的质量、可控性和多样性。我们提出了一种新的超网络调制的视觉文本编码方案来预测编码层的权重更新，从而实现从视觉信息（例如布局、内容）到相应的潜在域的有效传输。实验结果表明，我们的模型以额外的检索视觉数据为指导，优于现有的基于 GAN 的模型。在 COCO 数据集上，与最先进的方法相比，我们实现了更好的 FID 为 9.13，生成器参数减少了 3.5 倍。
  - [UPST-NeRF：用于 3D 场景的神经辐射场的通用逼真风格转移](https://arxiv.org/abs/2208.07059) | [***``[code]``***](https://github.com/semchan/UPST-NeRF)
    > 3D 场景逼真风格化旨在根据给定的风格图像从任意新颖的视图生成逼真的图像，同时确保从不同视点渲染时的一致性。现有的一些具有神经辐射场的风格化方法可以通过将风格图像的特征与多视图图像相结合来训练3D场景，从而有效地预测风格化场景。然而，这些方法会生成包含令人反感的伪影的新颖视图图像。此外，它们无法为 3D 场景实现通用的逼真风格化。因此，造型图像必须重新训练基于神经辐射场的 3D 场景表示网络。我们提出了一种新颖的 3D 场景逼真风格迁移框架来解决这些问题。它可以用 2D 风格的图像实现逼真的 3D 场景风格转换。我们首先预训练了一个 2D 真实感风格迁移网络，可以满足任何给定内容图像和风格图像之间的真实感风格迁移。然后，我们使用体素特征来优化 3D 场景并获得场景的几何表示。最后，我们共同优化了一个超网络，以实现任意风格图像的场景逼真风格迁移。在迁移阶段，我们使用预训练的 2D 真实感网络来约束 3D 场景中不同视图和不同风格图像的真实感风格。实验结果表明，我们的方法不仅实现了任意风格图像的 3D 逼真风格转换，而且在视觉质量和一致性方面优于现有方法。项目页面：此 https URL。
## Aug7 - Aug13, 2022
  - [HRF-Net：来自稀疏输入的整体辐射场](https://arxiv.org/abs/2208.04717) | [code]
    > 我们提出了 HRF-Net，这是一种基于整体辐射场的新型视图合成方法，它使用一组稀疏输入来渲染新颖的视图。最近的泛化视图合成方法也利用了辐射场，但渲染速度不是实时的。现有的方法可以有效地训练和渲染新颖的视图，但它们不能推广到看不见的场景。我们的方法解决了用于泛化视图合成的实时渲染问题，包括两个主要阶段：整体辐射场预测器和基于卷积的神经渲染器。这种架构不仅可以基于隐式神经场推断出一致的场景几何，还可以使用单个 GPU 有效地渲染新视图。我们首先在 DTU 数据集的多个 3D 场景上训练 HRF-Net，并且该网络可以仅使用光度损失对看不见的真实和合成数据产生似是而非的新颖视图。此外，我们的方法可以利用单个场景的一组更密集的参考图像来生成准确的新颖视图，而无需依赖额外的显式表示，并且仍然保持预训练模型的高速渲染。实验结果表明，HRF-Net 在各种合成和真实数据集上优于最先进的可泛化神经渲染方法。
## Jul31 - Aug6, 2022
  - [NeSF: 用于 3D 场景的可概括语义分割的神经语义场](https://research.google/pubs/pub51563/) | [code]
    > 我们提出了 NeSF，一种从预训练的密度场和稀疏的 2D 语义监督产生 3D 语义场的方法。我们的方法通过利用将 3D 信息存储在神经域中的神经表示来避开传统的场景表示。尽管仅由 2D 信号监督，我们的方法能够从新颖的相机姿势生成 3D 一致的语义图，并且可以在任意 3D 点进行查询。值得注意的是，NeSF 与任何产生密度场的方法兼容，并且随着预训练密度场质量的提高，其准确性也会提高。我们的实证分析证明了在令人信服的合成场景上与竞争性 2D 和 3D 语义分割基线相当的质量，同时还提供了现有方法无法提供的功能。
  - [Transformers as Meta-Learners for Implicit Neural Representations, ECCV2022](https://arxiv.org/abs/2208.02801) | [***``[code]``***](https://yinboc.github.io/trans-inr/)
    > 近年来，隐式神经表示 (INR) 已经出现并显示出其优于离散表示的优势。然而，将 INR 拟合到给定的观测值通常需要从头开始使用梯度下降进行优化，这是低效的，并且不能很好地泛化稀疏的观测值。为了解决这个问题，大多数先前的工作都训练了一个超网络，该超网络生成单个向量来调制 INR 权重，其中单个向量成为限制输出 INR 重建精度的信息瓶颈。最近的工作表明，通过基于梯度的元学习，可以在没有单向量瓶颈的情况下精确推断 INR 中的整个权重集。受基于梯度的元学习的广义公式的启发，我们提出了一个公式，该公式使用 Transformer 作为 INR 的超网络，它可以使用专门作为集合到集合映射的 Transformer 直接构建整个 INR 权重集。我们展示了我们的方法在不同任务和领域中构建 INR 的有效性，包括 2D 图像回归和 3D 对象的视图合成。我们的工作在 Transformer 超网络和基于梯度的元学习算法之间建立了联系，我们为理解生成的 INR 提供了进一步的分析。
  - [VolTeMorph：体积表示的实时、可控和可泛化动画](https://arxiv.org/pdf/2208.00949) | [code]
    > 最近，用于场景重建和新颖视图合成的体积表示越来越受欢迎，这使人们重新关注在高可见度下对体积内容进行动画处理质量和实时性。虽然基于学习函数的隐式变形方法可以产生令人印象深刻的结果，但它们对于艺术家和内容创作者来说是“黑匣子”，它们需要大量的训练数据才能进行有意义的概括，而且它们不会在训练数据之外产生现实的外推。在这项工作中，我们通过引入一种实时、易于使用现成软件进行编辑并且可以令人信服地推断的体积变形方法来解决这些问题。为了展示我们方法的多功能性，我们将其应用于两个场景：基于物理的对象变形和远程呈现，其中化身使用混合形状进行控制。我们还进行了彻底的实验，表明我们的方法优于结合隐式变形的体积方法和基于网格变形的方法。
## Jul24 - Jul30, 2022
  - [ZEPI-Net：通过内部跨尺度对极平面图像零样本学习的光场超分辨率, Neural Processing Letters (2022)](https://link.springer.com/article/10.1007/s11063-022-10955-x) | [code]
    > 光场 (LF) 成像的许多应用都受到空间角分辨率问题的限制，因此需要高效的超分辨率技术。最近，基于学习的解决方案比传统的超分辨率（SR）技术取得了显着更好的性能。不幸的是，学习或训练过程在很大程度上依赖于训练数据集，这对于大多数 LF 成像应用程序来说可能是有限的。在本文中，我们提出了一种基于零样本学习的新型 LF 空间角 SR 算法。我们建议在核平面图像 (EPI) 空间中学习跨尺度可重用特征，并避免显式建模场景先验或从大量 LF 中隐式学习。最重要的是，在不使用任何外部 LF 的情况下，所提出的算法可以同时在空间域和角域中超分辨 LF。此外，所提出的解决方案没有深度或视差估计，这通常由现有的 LF 空间和角度 SR 采用。通过使用一个简单的 8 层全卷积网络，我们表明所提出的算法可以产生与最先进的空间 SR 相当的结果。我们的算法在多组公共 LF 数据集上的角度 SR 方面优于现有方法。实验结果表明，跨尺度特征可以很好地学习并在 EPI 空间中用于 LF SR。
  - [ObjectFusion：具有神经对象先验的准确对象级 SLAM, Graphical Models, Volume 123, September 2022](https://www.sciencedirect.com/science/article/pii/S1524070322000418) | [code]
    > 以前的对象级同步定位和映射 (SLAM) 方法仍然无法以有效的方式创建高质量的面向对象的 3D 地图。主要挑战来自如何有效地表示对象形状以及如何将这种对象表示有效地应用于准确的在线相机跟踪。在本文中，我们提供 ObjectFusion 作为静态场景中的一种新颖的对象级 SLAM，它通过利用神经对象先验，有效地创建具有高质量对象重建的面向对象的 3D 地图。我们提出了一种仅具有单个编码器-解码器网络的神经对象表示，以有效地表达各种类别的对象形状，这有利于对象实例的高质量重建。更重要的是，我们建议将这种神经对象表示转换为精确测量，以共同优化对象形状、对象姿态和相机姿态，以实现最终准确的 3D 对象重建。通过对合成和真实世界 RGB-D 数据集的广泛评估，我们表明我们的 ObjectFusion 优于以前的方法，具有更好的对象重建质量，使用更少的内存占用，并且以更有效的方式，尤其是在对象级别。
  - [通过 NeRF Attention 进行端到端视图合成](https://arxiv.org/abs/2207.14741) | [code]
    > 在本文中，我们提出了一个用于视图合成的简单 seq2seq 公式，其中我们将一组光线点作为输入和输出与光线相对应的颜色。在这个 seq2seq 公式上直接应用标准转换器有两个限制。首先，标准注意力不能成功地适应体积渲染过程，因此合成视图中缺少高频分量。其次，将全局注意力应用于所有光线和像素是非常低效的。受神经辐射场 (NeRF) 的启发，我们提出了 NeRF 注意力 (NeRFA) 来解决上述问题。一方面，NeRFA 将体积渲染方程视为软特征调制过程。通过这种方式，特征调制增强了具有类似 NeRF 电感偏置的变压器。另一方面，NeRFA 执行多阶段注意力以减少计算开销。此外，NeRFA 模型采用光线和像素转换器来学习光线和像素之间的相互作用。 NeRFA 在四个数据集上展示了优于 NeRF 和 NerFormer 的性能：DeepVoxels、Blender、LLFF 和 CO3D。此外，NeRFA 在两种设置下建立了新的 state-of-the-art：单场景视图合成和以类别为中心的新颖视图合成。该代码将公开发布。
## Previous weeks
  - [CLA-NeRF：类别级关节神经辐射场, ICRA2022](https://arxiv.org/abs/2202.00181) | [code]
    > 我们提出了 CLA-NeRF——一种类别级的关节神经辐射场，可以执行视图合成、部分分割和关节姿态估计。 CLA-NeRF 在对象类别级别进行训练，不使用 CAD 模型和深度，而是使用一组具有地面实况相机姿势和部分片段的 RGB 图像。在推理过程中，只需对已知类别中未见过的 3D 对象实例进行少量 RGB 视图（即少镜头）即可推断对象部分分割和神经辐射场。给定一个关节姿态作为输入，CLA-NeRF 可以执行关节感知体积渲染，以在任何相机姿态下生成相应的 RGB 图像。此外，可以通过逆向渲染来估计对象的关节姿势。在我们的实验中，我们对合成数据和真实数据的五个类别的框架进行了评估。在所有情况下，我们的方法都显示了真实的变形结果和准确的关节姿态估计。我们相信，少量的关节对象渲染和关节姿势估计都为机器人感知和与看不见的关节对象交互打开了大门。
  - [GRAF：用于 3D 感知图像合成的生成辐射场, NeurIPS2020](https://avg.is.mpg.de/publications/schwarz2020NeurIPS) | [***``[code]``***](https://github.com/autonomousvision/graf)
    > 虽然 2D 生成对抗网络已经实现了高分辨率图像合成，但它们在很大程度上缺乏对 3D 世界和图像形成过程的理解。因此，它们不提供对相机视点或物体姿势的精确控制。为了解决这个问题，最近的几种方法将基于中间体素的表示与可微渲染相结合。然而，现有方法要么产生低图像分辨率，要么在解开相机和场景属性方面存在不足，例如，对象身份可能随视点而变化。在本文中，我们提出了一种辐射场的生成模型，该模型最近被证明在单个场景的新颖视图合成方面是成功的。与基于体素的表示相比，辐射场并不局限于 3D 空间的粗略离散化，还允许解开相机和场景属性，同时在存在重建模糊性的情况下优雅地退化。通过引入基于多尺度补丁的鉴别器，我们展示了高分辨率图像的合成，同时仅从未定位的 2D 图像训练我们的模型。我们系统地分析了我们在几个具有挑战性的合成和现实世界数据集上的方法。我们的实验表明，辐射场是生成图像合成的强大表示，可生成以高保真度渲染的 3D 一致模型。
  - [GRF：学习用于 3D 场景表示和渲染的一般辐射场, ICCV2021(oral)](https://arxiv.org/abs/2010.04595) | [***``[code]``***](https://github.com/alextrevithick/GRF)
    > 我们提出了一个简单而强大的神经网络，它仅从 2D 观察中隐式表示和渲染 3D 对象和场景。该网络将 3D 几何建模为一般辐射场，它以一组具有相机位姿和内在函数的 2D 图像作为输入，为 3D 空间的每个点构建内部表示，然后渲染该点的相应外观和几何观察从任意位置。我们方法的关键是学习 2D 图像中每个像素的局部特征，然后将这些特征投影到 3D 点，从而产生一般和丰富的点表示。我们还集成了一种注意力机制来聚合来自多个 2D 视图的像素特征，从而隐式考虑视觉遮挡。大量实验表明，我们的方法可以为新物体、看不见的类别和具有挑战性的现实世界场景生成高质量和逼真的新视图。
  - [pixelNeRF：来自一个或几个图像的神经辐射场, CVPR2021](https://arxiv.org/abs/2012.02190) | [***``[code]``***](https://github.com/sxyu/pixel-nerf)
    > 我们提出了 pixelNeRF，这是一种学习框架，可以预测以一个或几个输入图像为条件的连续神经场景表示。构建神经辐射场的现有方法涉及独立优化每个场景的表示，需要许多校准视图和大量计算时间。我们通过引入一种以完全卷积方式在图像输入上调节 NeRF 的架构，朝着解决这些缺点迈出了一步。这允许网络在多个场景中进行训练，以先学习一个场景，使其能够从一组稀疏的视图（少至一个）以前馈方式执行新颖的视图合成。利用 NeRF 的体积渲染方法，我们的模型可以直接从图像中训练，无需明确的 3D 监督。我们在 ShapeNet 基准上进行了广泛的实验，用于具有保留对象以及整个未见类别的单图像新颖视图合成任务。我们通过在多对象 ShapeNet 场景和来自 DTU 数据集的真实场景上展示 pixelNeRF 的灵活性，进一步展示了它的灵活性。在所有情况下，对于新颖的视图合成和单图像 3D 重建，pixelNeRF 都优于当前最先进的基线。有关视频和代码，请访问项目网站：此 https 网址
  - [用于优化基于坐标的神经表示的学习初始化, CVPR2021](https://www.matthewtancik.com/learnit) | [***``[code]``***](https://github.com/tancik/learnit)
    > 基于坐标的神经表示已显示出作为复杂低维信号的离散、基于数组的表示的替代方案的重要前景。然而，从每个新信号的随机初始化权重优化基于坐标的网络是低效的。我们建议应用标准的元学习算法来学习这些全连接网络的初始权重参数，这些参数基于所表示的底层信号类别（例如，面部图像或椅子的 3D 模型）。尽管只需要在实现中进行微小的更改，但使用这些学习到的初始权重可以在优化过程中实现更快的收敛，并且可以作为所建模信号类的强先验，从而在只有给定信号的部分观察可用时产生更好的泛化。我们在各种任务中探索这些好处，包括表示 2D 图像、重建 CT 扫描以及从 2D 图像观察中恢复 3D 形状和场景。
  - [pi-GAN：用于 3D 感知图像合成的周期性隐式生成对抗网络, CVPR2021(oral)](https://marcoamonteiro.github.io/pi-GAN-website/) | [***``[code]``***](https://github.com/marcoamonteiro/pi-GAN)
    > 我们见证了 3D 感知图像合成的快速进展，利用了生成视觉模型和神经渲染的最新进展。然而，现有方法在两个方面存在不足：首先，它们可能缺乏底层 3D 表示或依赖于视图不一致的渲染，因此合成的图像不是多视图一致的；其次，它们通常依赖于表达能力不足的表示网络架构，因此它们的结果缺乏图像质量。我们提出了一种新颖的生成模型，称为周期性隐式生成对抗网络（π-GAN 或 pi-GAN），用于高质量的 3D 感知图像合成。 π-GAN 利用具有周期性激活函数和体积渲染的神经表示将场景表示为具有精细细节的视图一致的 3D 表示。所提出的方法获得了具有多个真实和合成数据集的 3D 感知图像合成的最新结果。
  - [单张图像的人像神经辐射场](https://portrait-nerf.github.io/) | [code]
    > 我们提出了一种从单个爆头肖像估计神经辐射场 (NeRF) 的方法。虽然 NeRF 已经展示了高质量的视图合成，但它需要静态场景的多个图像，因此对于随意捕捉和移动主体是不切实际的。在这项工作中，我们建议使用使用灯光舞台肖像数据集的元学习框架来预训练多层感知器 (MLP) 的权重，该多层感知器隐含地对体积密度和颜色进行建模。为了提高对看不见的人脸的泛化能力，我们在由 3D 人脸可变形模型近似的规范坐标空间中训练 MLP。我们使用受控捕获对方法进行定量评估，并展示了对真实肖像图像的泛化性，显示出对最先进技术的有利结果。
  - [CAMPARI：相机感知分解生成神经辐射场](https://arxiv.org/pdf/2103.17269.pdf) | [code]
    > 深度生成模型的巨大进步导致了逼真的图像合成。在取得令人信服的结果的同时，大多数方法都在二维图像域中运行，而忽略了我们世界的三维性质。因此，最近的几项工作提出了具有 3D 感知能力的生成模型，即场景以 3D 建模，然后可微分地渲染到图像平面。这导致了令人印象深刻的 3D 一致性，但纳入这种偏差是有代价的：相机也需要建模。当前的方法假定固定的内在函数和预先定义的相机姿势范围。因此，实际数据通常需要参数调整，如果数据分布不匹配，结果会下降。我们的关键假设是，与图像生成器一起学习相机生成器会导致更原则性的 3D 感知图像合成方法。此外，我们建议将场景分解为背景和前景模型，从而实现更有效和更清晰的场景表示。在从原始的、未定型的图像集合中进行训练时，我们学习了一个 3D 和相机感知的生成模型，它不仅忠实地恢复了图像，而且还忠实地恢复了相机数据分布。在测试时，我们的模型生成的图像可以显式控制相机以及场景的形状和外观。
  - [NeRF-VAE：几何感知 3D 场景生成模型](https://arxiv.org/abs/2104.00587) | [code]
    > 我们提出了 NeRF-VAE，这是一种 3D 场景生成模型，它通过 NeRF 和可微体渲染结合了几何结构。与 NeRF 相比，我们的模型考虑了跨场景的共享结构，并且能够使用摊销推理推断新场景的结构——无需重新训练。 NeRF-VAE 的显式 3D 渲染过程进一步将先前的生成模型与缺乏几何结构的基于卷积的渲染进行对比。我们的模型是一个 VAE，它通过在潜在场景表示上调节辐射场来学习辐射场的分布。我们表明，经过训练，NeRF-VAE 能够使用很少的输入图像从以前看不见的 3D 环境中推断和渲染几何一致的场景。我们进一步证明了 NeRF-VAE 可以很好地推广到分布式相机，而卷积模型则不能。最后，我们介绍并研究了 NeRF-VAE 解码器的一种基于注意力的调节机制，该机制提高了模型性能。
  - [具有局部条件辐射场的无约束场景生成, ICCV2021](https://apple.github.io/ml-gsn/) | [***``[code]``***](https://github.com/apple/ml-gsn)
    > 我们遵循对抗性学习框架，其中生成器通过其辐射场对场景进行建模，鉴别器尝试区分从这些辐射场渲染的图像和真实场景的图像。从概念上讲，我们的模型将场景的辐射场分解为许多小的局部辐射场，这些辐射场是由二维潜在代码 W 网格上的条件产生的。W 可以解释为表示场景的潜在平面图。
  - [MVSNeRF：从多视图立体快速概括辐射场重建, ICCV2021](https://apchenstu.github.io/mvsnerf/) | [***``[code]``***](https://github.com/apchenstu/mvsnerf)
    > 我们提出了 MVSNeRF，一种新颖的神经渲染方法，可以有效地重建神经辐射场以进行视图合成。与先前的神经辐射场工作考虑对密集捕获的图像进行逐场景优化不同，我们提出了一个通用的深度神经网络，它可以通过快速网络推理仅从三个附近的输入视图重建辐射场。我们的方法利用平面扫描成本体积（广泛用于多视图立体）进行几何感知场景推理，并将其与基于物理的体积渲染相结合用于神经辐射场重建。我们在 DTU 数据集中的真实对象上训练我们的网络，并在三个不同的数据集上对其进行测试，以评估其有效性和普遍性。我们的方法可以跨场景（甚至是室内场景，与我们的对象训练场景完全不同）进行泛化，并仅使用三个输入图像生成逼真的视图合成结果，显着优于可泛化辐射场重建的并行工作。此外，如果捕捉到密集的图像，我们估计的辐射场表示可以很容易地进行微调；与 NeRF 相比，这导致具有更高渲染质量和更短优化时间的快速每场景重建。
  - [立体辐射场 (SRF)：从新场景的稀疏视图中学习视图合成, CVPR2021](https://arxiv.org/abs/2104.06935) | [***``[code]``***](https://virtualhumans.mpi-inf.mpg.de/srf/)
    > 最近的神经视图合成方法取得了令人印象深刻的质量和真实性，超越了依赖多视图重建的经典管道。最先进的方法，例如 NeRF，旨在使用神经网络学习单个场景，并且需要密集的多视图输入。在新场景上进行测试需要从头开始重新训练，这需要 2-3 天。在这项工作中，我们介绍了立体辐射场 (SRF)，这是一种端到端训练的神经视图合成方法，可以推广到新场景，并且在测试时只需要稀疏视图。核心思想是一种受经典多视图立体方法启发的神经架构，它通过在立体图像中找到相似的图像区域来估计表面点。在 SRF 中，我们预测每个 3D 点的颜色和密度，给定输入图像中立体对应的编码。编码是通过成对相似性的集合隐式学习的——模拟经典立体声。实验表明，SRF 在场景上学习结构而不是过度拟合。我们在 DTU 数据集的多个场景上进行训练，并在不重新训练的情况下推广到新场景，只需要 10 个稀疏和展开的视图作为输入。我们展示了 10-15 分钟的微调进一步改善了结果，与特定场景的模型相比，获得了更清晰、更详细的结果。代码、模型和视频可在此 https 网址上找到。
  - [用于遮挡感知的基于图像的渲染的神经射线, CVPR2022](https://liuyuan-pal.github.io/NeuRay/) | [***``[code]``***](https://github.com/liuyuan-pal/NeuRay)
    > 我们提出了一种新的神经表示，称为神经射线 (NeuRay)，用于新的视图合成任务。最近的工作从输入视图的图像特征构建辐射场以渲染新颖的视图图像，从而能够泛化到新场景。但是，由于遮挡，3D 点可能对某些输入视图不可见。在这样的 3D 点上，这些泛化方法将包括来自不可见视图的不一致图像特征，这会干扰辐射场的构建。为了解决这个问题，我们在 NeuRay 表示中预测 3D 点对输入视图的可见性。这种可见性使辐射场构建能够专注于可见图像特征，从而显着提高其渲染质量。同时，提出了一种新颖的一致性损失，以在对特定场景进行微调时改进 NeuRay 中的可见性。实验表明，我们的方法在推广到看不见的场景时在新颖的视图合成任务上实现了最先进的性能，并且在微调后优于每个场景的优化方法。
  - [节食 NeRF：语义一致的 Few-Shot 视图合成, ICCV2021](https://www.ajayj.com/dietnerf) | [***``[code]``***](https://github.com/ajayjain/DietNeRF)
    > 我们提出了 DietNeRF，一种从几张图像估计的 3D 神经场景表示。神经辐射场 (NeRF) 通过多视图一致性学习场景的连续体积表示，并且可以通过光线投射从新颖的视点进行渲染。虽然 NeRF 在给定许多图像的情况下具有令人印象深刻的重建几何和精细细节的能力，对于具有挑战性的 360° 场景最多可重建 100 个，但当只有少数输入视图可用时，它通常会为其图像重建目标找到退化的解决方案。为了提高few-shot质量，我们提出了DietNeRF。我们引入了一种辅助语义一致性损失，它鼓励以新颖的姿势进行逼真的渲染。 DietNeRF 在单个场景上进行训练，以 (1) 从相同的姿势正确渲染给定的输入视图，以及 (2) 在不同的随机姿势中匹配高级语义属性。我们的语义损失使我们能够从任意姿势监督 DietNeRF。我们使用预训练的视觉编码器提取这些语义，例如 CLIP，这是一种视觉转换器，通过自然语言监督从网络挖掘出的数亿张不同的单视图 2D 照片进行训练。在实验中，DietNeRF 在从头开始学习时提高了少镜头视图合成的感知质量，在多视图数据集上进行预训练时，可以用少至一张观察到的图像渲染新视图，并生成完全未观察到的区域的合理完成。
  - [CodeNeRF：对象类别的解开神经辐射场, ICCV2021(oral)](https://www.google.com/url?q=https%3A%2F%2Farxiv.org%2Fpdf%2F2109.01750.pdf&sa=D&sntz=1&usg=AOvVaw1Fnir0e4aRa22Nt0HoXDWh) | [***``[code]``***](https://www.google.com/url?q=https%3A%2F%2Fgithub.com%2Fwbjang%2Fcode-nerf&sa=D&sntz=1&usg=AOvVaw2eD5ZoRbk2aWFuwUSHlh5_)
    > CodeNeRF 是一种隐式 3D 神经表示，它学习对象形状和纹理在一个类别中的变化，并且可以从一组姿势图像中进行训练，以合成看不见的对象的新视图。与特定场景的原始 NeRF 不同，CodeNeRF 通过学习单独的嵌入来学习解开形状和纹理。在测试时，给定一个看不见的物体的单个未定位图像，CodeNeRF 通过优化联合估计相机视点、形状和外观代码。看不见的物体可以从单个图像中重建，然后从新的视点渲染，或者通过改变潜在代码编辑它们的形状和纹理。我们在 SRN 基准上进行了实验，结果表明 CodeNeRF 可以很好地泛化到看不见的对象，并且在测试时需要已知相机姿态的方法达到同等性能。我们在真实世界图像上的结果表明，CodeNeRF 可以弥合模拟到真实的差距。
  - [StyleNeRF：用于高分辨率图像合成的基于样式的 3D 感知生成器, ICLR2022](https://jiataogu.me/style_nerf/) | [***``[code]``***](https://github.com/facebookresearch/StyleNeRF)
    > 我们提出了 StyleNeRF，这是一种 3D 感知生成模型，用于具有高多视图一致性的逼真的高分辨率图像合成，可以在非结构化 2D 图像上进行训练。现有方法要么无法合成具有精细细节的高分辨率图像，要么产生明显的 3D 不一致伪影。此外，他们中的许多人缺乏对风格属性和明确的 3D 相机姿势的控制。 StyleNeRF 将神经辐射场 (NeRF) 集成到基于样式的生成器中，以应对上述挑战，即提高渲染效率和 3D 一致性以生成高分辨率图像。我们执行体积渲染只是为了生成一个低分辨率的特征图，并在 2D 中逐步应用上采样来解决第一个问题。为了减轻 2D 上采样引起的不一致性，我们提出了多种设计，包括更好的上采样器和新的正则化损失。通过这些设计，StyleNeRF 可以以交互速率合成高分辨率图像，同时保持高质量的 3D 一致性。 StyleNeRF 还可以控制相机姿势和不同级别的样式，可以推广到看不见的视图。它还支持具有挑战性的任务，包括放大和缩小、样式混合、反转和语义编辑。
  - [GNeRF：基于 GAN 的无姿势相机的神经辐射场, ICCV2021(oral)](https://arxiv.org/abs/2103.15606) | [code]
    > 我们介绍了 GNeRF，这是一个将生成对抗网络 (GAN) 与神经辐射场 (NeRF) 重建相结合的框架，用于具有未知甚至随机初始化相机姿势的复杂场景。最近基于 NeRF 的进展因显着的逼真的新视图合成而受到欢迎。然而，它们中的大多数严重依赖于准确的相机位姿估计，而最近的一些方法只能在相机轨迹相对较短的大致前向场景中优化未知相机位姿，并且需要粗略的相机位姿初始化。不同的是，我们的 GNeRF 仅将随机初始化的姿势用于复杂的由外而内的场景。我们提出了一种新颖的两阶段端到端框架。第一阶段将 GAN 的使用带入新领域，以联合优化粗略的相机姿势和辐射场，而第二阶段通过额外的光度损失对它们进行细化。我们使用混合迭代优化方案克服了局部最小值。对各种合成和自然场景的广泛实验证明了 GNeRF 的有效性。更令人印象深刻的是，我们的方法在那些以前被认为极具挑战性的重复模式甚至低纹理的场景中优于基线。
  - [NeRD：来自图像集合的神经反射分解, ICCV2021](https://markboss.me/publication/2021-nerd/#:~:text=NeRD%20is%20a%20novel%20method,can%20turn%20around%20the%20object.) | [***``[code]``***](https://github.com/cgtuebingen/NeRD-Neural-Reflectance-Decomposition)
    > 将场景分解为其形状、反射率和照明度是计算机视觉和图形学中一个具有挑战性但重要的问题。当照明不是实验室条件下的单一光源而是不受约束的环境照明时，这个问题本质上更具挑战性。尽管最近的工作表明可以使用隐式表示来模拟物体的辐射场，但这些技术中的大多数只能实现视图合成而不是重新照明。此外，评估这些辐射场是资源和时间密集型的。我们提出了一种神经反射分解 (NeRD) 技术，该技术使用基于物理的渲染将场景分解为空间变化的 BRDF 材料属性。与现有技术相比，我们的输入图像可以在不同的照明条件下捕获。此外，我们还提出了将学习到的反射体积转换为可重新照明的纹理网格的技术，从而能够使用新颖的照明进行快速实时渲染。我们通过在合成数据集和真实数据集上的实验证明了所提出方法的潜力，我们能够从图像集合中获得高质量的可重新点亮的 3D 资产。
  - [NeRF++：分析和改进神经辐射场](https://arxiv.org/abs/2010.07492) | [***``[code]``***](https://github.com/Kai-46/nerfplusplus;)
    > 神经辐射场 (NeRF) 为各种捕捉设置实现了令人印象深刻的视图合成结果，包括有界场景的 360 度捕捉以及有界和无界场景的前向捕捉。 NeRF 将表示视图不变不透明度和视图相关颜色体积的多层感知器 (MLP) 拟合到一组训练图像，并基于体积渲染技术对新视图进行采样。在这份技术报告中，我们首先评论了辐射场及其潜在的模糊性，即形状-辐射模糊度，并分析了 NeRF 在避免这种模糊性方面的成功。其次，我们解决了将 NeRF 应用于大规模、无界 3D 场景中对象的 360 度捕获所涉及的参数化问题。我们的方法在这种具有挑战性的场景中提高了视图合成保真度。此 https 网址提供了代码。
  - [GIRAFFE：将场景表示为合成生成神经特征场, CVPR2021(oral)](https://arxiv.org/abs/2011.12100) | [***``[code]``***](https://github.com/autonomousvision/giraffe)
    > 深度生成模型允许以高分辨率进行逼真的图像合成。但对于许多应用程序来说，这还不够：内容创建还需要可控。虽然最近的几项工作研究了如何解开数据变化的潜在因素，但它们中的大多数都在 2D 中运行，因此忽略了我们的世界是 3D 的。此外，只有少数作品考虑场景的构图性质。我们的关键假设是，将合成 3D 场景表示合并到生成模型中会导致更可控的图像合成。将场景表示为合成生成神经特征场使我们能够从背景中解开一个或多个对象以及单个对象的形状和外观，同时从非结构化和未定型的图像集合中学习，而无需任何额外的监督。将这种场景表示与神经渲染管道相结合，可以生成快速且逼真的图像合成模型。正如我们的实验所证明的那样，我们的模型能够解开单个对象，并允许在场景中平移和旋转它们以及改变相机姿势。
  - [Fig-NeRF：用于 3D 对象类别建模的图地面神经辐射场, 3DV2021](https://fig-nerf.github.io/) | [code]
    > 我们研究使用神经辐射场 (NeRF) 从输入图像的集合中学习高质量的 3D 对象类别模型。与以前的工作相比，我们能够做到这一点，同时将前景对象与不同的背景分开。我们通过 2 分量 NeRF 模型 FiG-NeRF 实现了这一点，该模型更喜欢将场景解释为几何恒定的背景和代表对象类别的可变形前景。我们表明，这种方法可以仅使用光度监督和随意捕获的对象图像来学习准确的 3D 对象类别模型。此外，我们的两部分分解允许模型执行准确和清晰的模态分割。我们使用合成的、实验室捕获的和野外数据，通过视图合成和图像保真度指标对我们的方法进行定量评估。我们的结果证明了令人信服的 3D 对象类别建模，其性能超过了现有方法的性能。
  - [NerfingMVS：室内多视角立体神经辐射场的引导优化, ICCV2021(oral)](https://arxiv.org/abs/2109.01129) | [***``[code]``***](https://github.com/weiyithu/NerfingMVS)
    > 在这项工作中，我们提出了一种新的多视图深度估计方法，该方法在最近提出的神经辐射场 (NeRF) 上利用了传统的 SfM 重建和基于学习的先验。与现有的依赖于估计对应的基于神经网络的优化方法不同，我们的方法直接优化隐式体积，消除了在室内场景中匹配像素的挑战性步骤。我们方法的关键是利用基于学习的先验来指导 NeRF 的优化过程。我们的系统首先通过微调其稀疏 SfM 重建来适应目标场景上的单目深度网络。然后，我们证明了 NeRF 的形状-辐射模糊性仍然存在于室内环境中，并建议通过采用适应的深度先验来监控体绘制的采样过程来解决这个问题。最后，通过对渲染图像进行误差计算获得的每像素置信度图可用于进一步提高深度质量。实验表明，我们提出的框架在室内场景中显着优于最先进的方法，在基于对应的优化和基于 NeRF 的优化对适应深度先验的有效性方面提出了令人惊讶的发现。此外，我们表明引导优化方案不会牺牲神经辐射场的原始合成能力，提高了可见视图和新视图的渲染质量。