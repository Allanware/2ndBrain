# Coupling a recurrent neural network to SPAD TCSPC systems for real‐time fluorescence lifetime imaging

## Time Correlated Single Photon Counting (TCSPC)
A sample is excited periodically by a very short pulse of light. The molecules in the sample absorb this light and then emit fluorescence. The emitted photons are detected one at a time by a highly sensitive detector. These detectors are capable of detecting individual photons and registering their arrival times (time-tagged). Therefore, it can repeatedly measure the time delay between the excitation pulse and the arrival of the emitted photon at the detector. It builds a histogram of photon counts versus time over many cycles of excitation and detection that accurately represent the fluorescence decay profile of the sample.

### Traditional TCSPC setup
The instrumentation of a typical TCPSC FLI system features a **confocal** setup, including a **single-photon detector**, a dedicated TCSPC module for **time tagging**, and a PC for **lifetime estimation**. 
- Confocal: both the illumination and the detection are focused ("co-focused") on the same small volume within the sample. Light from parts of the sample that are not in the focal plane (above or below it) is mostly not detected because it is not focused on the pinhole and thus is blocked.
- single-photon detector + time tagging: SPAD, CMOS SPAD
- lifetime estimation

### Lifetime Estimation
#### Fitting Methods: 
Eg: Least Square, MLE
Pros: Accurate 
Cons: time-consuming, computationally expensive, not photon-efficient

#### Non-fitting methods
Eg: CMM, [Existing Neural Nets](#Existing%20Neural%20Nets)

CMM Pros: fast, can be implemented in real-time Fli systems (at the edge) => photon-efficient
CMM Cons: sensitive to noise, use of truncated histogram 

#### Existing Neural Nets 
E.g. CNN, ANN, GAN
Pros: 
- has the ability to resolve multi-exponential decays and achieve accurate and fast estimation even in low photon-count scenarios
- can extract high-level features and can be integrated into a large-scale neural network for end-to-end lifetime image analysis (segmentation and classification)
Cons: 
- at software level: not real-time
- still use histograms as input
- only exploration, not actual deployment of on-FPGA implementations

To resolve the Cons, they proposed [RNN on board](#RNN%20on%20board)

## RNN on board
Eliminate histogramming and process raw timestamps on the fly in an event-driven way. This approach enables the continuous and incremental updating of lifetime estimations with each incoming photon, so the lifetime can be read out during or right after the acquisition, reducing the data transfer bandwidth. 

Also, as an added benefit, since it does not take the whole histogram as input, the number of parameters the RNN require is mere hundreds, making it feasible to be deployed on board.

### Training 
#### Synthetic datasets for training and testing
lifetime assumed: 0.2 to 5 ns
laser repetition frequency assumed: 20 MHz.
Datasets with background noise (uniform noise) and without
#### Architecture
Vanila RNN, LSTM, GRU with 8, 16, and 32 hidden units (9 configs)

#### Objectives
RMSE, MAE, MAPE? I guess, since those are the ones used to evaluate the performance and be compared across architecture and with LS and CMM

### Evaluation
#### Cramer‐rao lower bound analysis
#### Performance on experimental dataset
The sample contains a mixture of fluorescent beads with three different lifetimes. Most pixels are assumed to contain mono-exponential fluorophores.
## Caveats 
The synthetic dataset assumes mono-exponential per pixel in a range of "common" lifetime values with a certain range of photon counts. 
Why L-S is always low 
## Do we need this 
Real Time Estimation? Not really
Reduction of the required hardware resources? No
Higher photon efficiency? No 
works better in low-photon-count scenario? Yes. But this model doesn't really address that.
Plus, we need the network to fit biexponential curve

# Generative adversarial network enables rapid and robust fluorescence lifetime image analysis in live cells

Deep learning algorithms may further improve the reliability in analyzing the low-photon-count (100–200 photon counts per pixel) or even ultralow-photon-count data (50–100 photon counts per pixel) for live-cell imaging.

I did not go with this paper b/c we already had ROI summing and their code was removed from Github (why?)

# Nellie: Automated organelle segmentation, tracking,  and hierarchical feature extraction in 2D/3D live-cell microscopy

## Motivation 
The dynamics of organelles such as mitochondrial lies at the center of cellular physiology (study of function) and pathology. However, the dynamic morphology and motility of these organelles, coupled with limitations inherent to microscopy pose significant challenges in feature extraction => manual, organelle-specific pipelines. 

A general analytical tool capable of providing detailed extraction of spatial and temporal features at multiple organelle scales that addresses:
- sufficient segmentation accuracy for dim or small objects
- limitations in tracking algorithms
- rely on the assumption that an organelle is a single and temporally consistent entity

## Features 
Nellie is able to segment and hierarchically divide organelles into logical subcomponents. These subcomponents are interrogated to produce motion capture markers that create linkages between adjacent frames, providing sub-voxel tracking capabilities.

They incorporate and introduce a multitude of both standard and advanced quantification techniques to extract a *hierarchical* pool of descriptive multi-level *spatial* and *temporal* features to choose from.

They show how one can use Nellie’s extracted features to unmix multiple organelle types from a single channel of a fluorescence timelapse. Second, they develop a novel multi-mesh approach to organelle graph construction, they use this multi-mesh to train an unsupervised graph autoencoder, and use the model to compare mitochondrial networks across a complex feature-space.

## Pipeline
### Preprocessing: structure contrast enhancement
Input: multi-dimensional image data, anything from 2D (YX) to 5D (CTZYX), of fluorescently labeled organelles. 
a multi-scale (for each different resolutions) modified Frangi filter is implemented:
1. the gamma parameter for the Frangi filter is derived from the minimum of the triangle and Otsu threshold values (size/scale-adaptive)

### Hierarchical deconstruction allows for multi-level organelle instance segmentation
1. instance segmentations of spatially disconnected organelles
2. skeletonization - deconstruct the organelle network into individually labeled branches
3. individual skeleton nodes

### Motion capture markers are generated for downstream tracking
- Independent Marker Generation
- use distance transformation and peak-identification to find markers 
- together with LoG to enable multi-scale

### MoCap Markers are connected
1. **Feature Vector Construction**: For each mocap marker, a feature vector is created to encapsulate the local characteristics and dynamics of the organelles. This includes:
	- The distance-transformed value (indicative of the organelle's radius) multiplied by two to determine the bounding box for each marker.
	- Computation of mean and variance for both the raw and preprocessed images within these bounding boxes, forming the 'stats vector'.
	- Calculation of the first six 2D Hu moment invariants (for translation, scale, and rotation invariance) in different projections, creating the 'Hu vectors'.

2. **Cost Matrix for Linking Markers**: To link markers between adjacent frames:
    - A 'speed matrix' is created by measuring the displacement of markers between frames and normalizing this by the time between frames and a maximum permissible speed.
    - Distance, stats, and Hu matrices are computed, normalized, and summed to form a final cost matrix. This matrix is used to determine the best linkage of mocap markers across frames based on their movement and changes in properties.
### Motion capture markers are tracked and used as guides for sub-voxel flow interpolation
- can be used to match voxels between timepoints and use voxels as track.

### Multi-scale feature extraction
- At the single-voxel level, such as fluorescence and structural intensity values from the raw and preprocessed images, and flow vectors: angular and linear velocity and acceleration, and directionality of each voxel
- At the single-node level, which are centered around individual skeleton voxels, Nellie calculates the thickness of the organelle at that node, the divergence, convergence end vergere (sum of the two) of its corresponding voxels’ flow vectors to the node center, as well as linear and angular magnitude and orientation flow vector variability metrics. 
- At the single-branch level, Nellie calculates skeleton-specific features, such as the length, thickness, aspect ratio, and tortuosity of the branch’s skeleton, but also features such as the branch’s corresponding segmentation mask’s area, major and minor axis length, extent, and solidity
- At the single-organelle level, Nellie calculates standard region properties, such as the area of the organelle, its major and minor axis length, its extent, and its solidity
- The organellar landscape as a whole represents the highest level of our hierarchy, containing aggregate information from all levels below it.
- Each hierarchical level can aggregate metrics from its lower level’s component

## Case Study
### unmixing of multiple organelle types in a single channel
Nellie + Random forest:
3 RF models: motility features alone, morphology features alone, or a combination of both

### learning of comparable variable-range latent space representations of organelle graphs
By transforming skeletonized networks of organelle segmentation masks into graph structures and utilizing a graph autoencoder to transform Nellie’s extensive feature outputs into a comparable representation,

the nodes of our graph as skeleton voxels underlying the organelle segmentation masks, with each node encapsulating features of the adjacent organelle voxels.

examining mitochondrial networks in cells treated with Ionomycin: train the encoder and decoder on data from different time point before and after treatment

Post-training, we deployed the model’s encoder to obtain latent space representations of each node across different timepoints.