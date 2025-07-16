# 4. Methodology

## 4.1. Overview

Given an input image $I\in\mathbb{R}^{H\times W\times3}$, where H and W denote the height and width respectively, we extract multiple low-level information $C=\{C_{1},C_{2},...,C_{M}\}$, where each $C_{i}\in\mathbb{R}^{H\times W\times3}$, $i=1,2,3,...,m$. Following UniFD \[36\], our approach uses the CLIP's visual backbone(ViT-L/14) in Fig. 2. To enable the model, pretrained on high-level images, to accept various low-level information inputs and ensure effective integration, while avoiding insufficient fusion either in the early or late stages, we transform the original transformer layer into a Cross-Low-level Expert LORA Transformer Layer, which will be introduced in Sec. 4.2. Furthermore, to prevent the loss of low-level input characteristics in deep transformer modeling, we employ a low-level information interaction adapter. The adapter further injection low-level information into the ViT for enhanced interaction, as discussed in Sec. 4.3. Finally, to select suitable features for different types of forgeries, we propose the Dynamic Feature Selection method to choose the most appropriate low-level features for the current type of forgery, which will be detailed in Sec. 4.4. The overall training phase of our framework will be presented in Sec. 4.5.

## 4.2. Cross-Low-level Transformer Layer

In our approach, we avoid merging features using straightforward fusion techniques. Instead, we strive to preserve the unique characteristics of each low-level information while capturing the interactions and influences between them. For the $M+1$ different low-level inputs with the high-level image input I denoted as $C_{0}$ and added to the set $C,C_{j}$, $(j=0,1,2,...,M)$, the visual encoder initially transforms the input tensors of size $\mathbb{R}^{H\times W\times3}$ into D-dimensional image features $F_{0}^{j}\in\mathbb{R}^{(1+L)\times D}$, where 1 represents the CLS token of the image, and $L=\frac{H\times W}{P^{2}}$ with P representing the number of patches. The input features for the $j^{th}$ information $C_{j}$ through the $i^{th}$ transformer layer are denoted as $F_{i}^{j}\in\mathbb{R}^{(1+L)\times D}$, $i=0,1,2,...,N$, where N denotes the number of layers in the transformer. The transformer module takes the patch-embedded features $F_{0}^{j}$ as input for each low-level information.

Considering the distinctiveness of each information, we aim to embed the knowledge of each information into the CLIP visual backbone without affecting the original pretrained weights. We employ the fine-tuning technique known as Lora \[17\], which is widely used in large language models and diffusion models, to incorporate modal knowledge through an additional plug-and-play module.

Each expert layer consists of our designed Multi-Lora-Expert Layer in Fig. 2(a), Self-Attention, residual connections, Layer Normalization and a FFN layer. In the Multi-Lora-Expert Layer at layer i, we employ Lora to process features specific to each input by designing different Lora experts. The computation is as follows:

$$
\hat{F}_{i}^{(j)}=W_{qkv}\cdot F_{i}^{(j)}+\frac{\alpha}{r}\Delta W_{j}\cdot F_{i}^{(j)}=W_{qkv}\cdot F_{i}^{(j)}+\frac{\alpha}{r}B_{j}A_{j}\cdot F_{i}^{(j)}
$$

(1)

Here, $\hat{F}_{\cdot}^{(j)}$ represents the output of $F_{i}^{(j)}$ after processing by the $j^{th}$ Lora expert and we set $r=4$ and $\alpha=8.$, $W_{qkv}$ denotes the matrix weights of the qkv in the attention layer and $\Delta W_{j}=B_{j}A_{j}$ is the trainable parameter of the $j^{th}$ Lora expert. Next, $\hat{F}_{i}^{(j)}$ serves as the input for the self-attention Q, K, V in the original CLIP, and the output after the FFN layer is denoted as $\overline{F}_{i}^{(j)}$. Noting that the features of each information are computed in parallel without interaction, we employ a cross-attention layer in the original output section to facilitate interaction between modalities, as computed by:

$$
F_{i+1}=\overline{F}_{i}+\beta_{i}MHA(LN(\overline{F}_{i}),LN(\overline{F}_{i}),LN(\overline{F}_{i}))
$$

(2)

Here, $LN(\cdot)$ represents LayerNorm, and the attention layer MHA() is suggested to use a multi-head attention mechanism with the number of heads set to 4. Furthermore, we apply a learnable vector $\beta_{i}\in\mathbb{R}^{(1+L)\times D}$ to balance the output of the attention layer with the input features, initially set to 0. This initialization strategy ensures that the unique features of each modality do not undergo drastic changes due to the injection of features from other modalities and adaptively integrates features related to forgery types contained in other modalities.

## 4.3. Low-level Information Interaction Adapter

Many work \[38, 57, 60\] suggests that the deeper layers of transformers might lead to the loss of low-level information, focusing instead on the learning of semantic information. Inspired by \[4\], to prevent our framework from losing critical classification features related to forgery types during the fusion of low-level information, we introduce a low-level information interaction adapter. This adapter is designed to capture low-level information priors and to enhance the significance of low-level information within the backbone. It operates parallel to the patch embedding layer of the CLIP image encoder and does not alter the architecture of the CLIP visual backbone. Unlike the vit-adapter \[4\], which injects spatial priors, our adapter injects low-level priors.

As illustrated, we utilize the first two blocks of ResNet50 \[16\], followed by global pooling and several $1\times1$ convolutions applied at the end to project the low-level information $C_{1},C_{2},...,C_{M}$ into D dimensions. Through this process, we obtain the feature vector $G_{0}\in\mathbb{R}^{D}$ extracted from the low-level encoder. To better integrate our features into the backbone, we design a cross-attention-based low-level feature injector and a low-level feature extractor.

**Low-level Feature Injector.** This module is used to inject low-level priors into the ViT. As shown in Fig. 2(b), for the output from each modality feature of the $i^{th}$ layer of CLIP using ViT-L, the features are concatenated into a feature vector $F_{i}\in\mathbb{R}^{(1+M)\cdot(1+L)\times D}$ which serves as the query for computing cross-attention. The low-level feature $G_{i}$ acts as the key and value in injecting into the modal feature $F_{i}$, represented by the following equation:

$$
F_{i}^{'}=F_{i}+\gamma_{i}MHA(LN(F_{i}), LN(G_{i}), LN(G_{i}))
$$

(3)

As before, LN and MHA operations respectively represent LayerNorm and multi-head attention mechanisms, with the number of heads set to 4. Similarly, we use a learnable vector $\gamma_{i}\in\mathbb{R}^{D}$ to balance the two different features.

**Modal Feature Extractor.** After injecting the low-level priors into the backbone, we perform the forward propagation process. We concatenate the output of each modality feature of the $(i+1)^{th}$ layer to obtain the feature vector $F_{i+1}$ and then apply a module composed of cross-attention and FFN to extract modal features, as shown in Fig. 2(b). This process is represented by the following equations:

$$
\overline{G_{i}}=G_{i}+\eta_{i}MHA(LN(G_{i}),LN(F_{i+1}),LN(F_{i+1}))
$$

(4)

$$
G_{i+1}=\tilde{G}_{i}+FFN(LN(\tilde{G_{i}}))
$$

(5)

Here, the low-level feature $G_{i}\in\mathbb{R}^{D}$ serves as the query, and the output $F_{i+1}\in\mathbb{R}^{(1+M)\cdot(1+L)\times L}$ from backbone acts as the key and value. Similar to the low-level feature injector, we use a learnable vector $\eta_{i}\in\mathbb{R}^{D}$ to balance the two different features. $G_{i+1}$ is then used as the input for the next low-level feature injector.

## 4.4. Dynamic Feature Selection

As mentioned in the introduction, since different features are often sensitive to different types of forgeries, simple feature concatenation or averaging followed by training with a unified classification head might lose some feature's advantages for detecting certain types of forgeries. To better integrate low-level features for generalizing to various forgery type detections, inspired by the mixed experts routing dynamic feature selection \[45\], we introduce a dynamic modal feature selection mechanism at the final output classification feature part of the model. Specifically, we extract the cls tokens of the final output, concatenate them, and denote this as $F_{cls}\in\mathbb{R}^{(1+M)\cdot D}$, which serves as the input for the dynamic router. The dynamic router employs a learnable fully connected neural network, with its matrix parameter defined as $W_{Router}\in\mathbb{R}^{(1+M)\cdot D\times(1+M)}$. The probability distribution for selecting each modal feature is computed as follows:

$$
p=SoftMax(W_{Router}F_{cls})
$$

(6)

For each feature, a corresponding classification head head, $i=0,1,2,...,M$ is prepared. The final classification result $\hat{P}(y)$ is obtained through the following equation:

$$
\hat{P}(y)=\sum_{i=0}^{M}p_{i}\cdot head_{i}(F_{cls}^{i})
$$

(7)

Here, $F_{cls}^{i}$ represents the cls token of the $i^{th}$ output feature. By adaptively learning a dynamic modal feature selection module, we enable the selection of suitable features for integration, thus allowing the classification to be tailored to the forgery type of the current image under detection. To balance the selection of different experts, we use entropy regularization loss as an additional constraint, as shown below:

$$
\mathcal{L}_{moe}=-\sum_{i=0}^{M}p_{i}log~p_{i}
$$

(8)

## 4.5. Training phase

We first train Lora Expert and the low-level information encoder for each type of low-level information and the high-level image information to ensure that the model learns knowledge relevant to AI-Generated image detection from both low-level and high-level information. Let the true label be y and the model's prediction be $\hat{P}(y)$ The training is performed using the cross-entropy loss as defined in Eq.9. Subsequently, we load these pre-trained weights into our framework and further train our carefully designed fusion module to ensure the adequate and appropriate fusion of each type of low-level and high-level information. Our final fused prediction results are given in Eq.7, and we optimize our overall framework using Eq. 10 as well, the loss is composed of the classification loss (Eq.9) and the expert balance regularization loss (Eq.8) weighted together. In our experiments, we set $\lambda=0.1.$

$$
\mathcal{L}_{cls}=-y\cdot log\hat{P}(y)-(1-y)\cdot log(1-\hat{P}(y))
$$

(9)

$$
\mathcal{L}_{total}=\mathcal{L}_{cls}+\lambda\mathcal{L}_{moe}
$$

(10)