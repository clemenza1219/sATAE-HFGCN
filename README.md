# sATAE-HFGCN
 This is the official implementation of our JBHI paper "Shared Attention-based Autoencoder with Hierarchical Fusion-based Graph Convolution Network for sEEG SOZ Identification".
## Abstract
Diagnosing seizure onset zone (SOZ) is a challenge in neurosurgery, where stereoelectroencephalography (sEEG) serves as a critical technique. 
In sEEG SOZ identification, the existing studies focus solely on the intra-patient representation of epileptic information, overlooking the general features of epilepsy across patients and feature interdependencies between feature elements in each contact site.
In order to address the aforementioned challenges, we propose the shared attention-based autoencoder (sATAE). 
sATAE is trained by sEEG data across all patients, with attention blocks introduced to enhance the representation of interdependencies between feature elements. 
Considering the spatial diversity of sEEG across patients, we introduce graph-based method for identification SOZ of each patient. 
However, the current graph-based methods for sEEG SOZ identification rely exclusively on static graphs to model epileptic networks.
Inspired by the finding of neuroscience that epileptic network is intricately characterized by the interplay of sophisticated equilibrium between fluctuating and stable states, we design the hierarchical fusion-based graph convolution network (HFGCN) to identify the SOZ.
HFGCN integrates the dynamic and static characteristics of epileptic networks through hierarchical weighting across different hierarchies, facilitating a more comprehensive learning of epileptic features and enriching node information for sEEG SOZ identification.
Combining sATAE and HFGCN, we perform comprehensive experiments with sATAE-HFGCN on the self-build sEEG dataset, which includes sEEG data from 17 patients with temporal lobe epilepsy.
The results show that our method, sATAE-HFGCN, not only achieves superior performance for identifying the SOZ of each patient, but also exhibits notable interpretability and generalizability, providing a novel and effective method for sEEG-based SOZ identification.
