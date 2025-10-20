This repository contains the work conducted during my summer internship at UQAM, in the first year of my master’s degree. <br/>
The code is applied to the HAM10000 dataset, available here: https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000 <br/>
<br/>
The project focuses on developing a pipeline to extract and align the shapes of moles in dermoscopic images using a tailored mathematical representation.**For details, see p.8-10 of my report.**<br/>
We then used these shape descriptors to train a Deep Learning model (an MLP) to discriminate between ‘nv’ (naevus: regular, benign moles) and ‘mel’ (melanomas: the deadliest form of skin cancer), based solely on shape information.<br/>

Finally, we compared this feature extraction pipeline with the current state-of-the-art deep learning approach: Convolutional Neural Networks (CNNs).<br/>

- **preprocessing** contains the code to segment the moles and obtain their shape-based representation.<br/>
- **modeling** contains the code used to build and compare the pixel-based approach to the shape-based approach, using a nested CV.<br/>
- **evaluation** contains the notebook used to build confidence intervals and perform statistical tests on our models, based on a bootstrapping approach. <br/>
