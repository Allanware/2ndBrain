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