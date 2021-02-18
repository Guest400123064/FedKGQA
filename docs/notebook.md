# Notebook

## Foundational Concepts

* Flavors of Federated Learning (FL)
  * Horizontal FL
    * Sample-Partitioned FL
    * Main Focus in this note
  * Vertical FL
  * Federated Transfer Learning
* Secure ML & Privacy-Preserving ML (PPML)
  * Secure ML is for **Integrity & Availability**
  * PPML is for **Confidentiality & Privacy**
* Attacker Models
  * Semi-honest (honest-but-curious)
  * Malicious
* Canonical Attacks Against ML Models
  * Reconstruction Attacks
  * Model Inversion Attacks
  * Membership-Inference Attacks
  * Attribute-Inference Attacks
* Essential PPML Technologies
  * Secured Multi-Party Computation (SMPC)
    * Categories
      * Oblivious Transfer (OT)
      * Secret Sharing (SS)
      * Threshold Homomorphic Encryption (THE)
  * Homomorphic Encryption (HE)
    * Note
      * HE is extremely **computationally expensive** so it is **NOT** suitable for model training
    * Categories
      * Partially (PHE)
      * Somewhat (SHE)
      * Fully (FHE)
  * Differential Privacy (DP) Mechanisms
    * Note
      * DP is merely a **criteria** to evaluate if a **mechanism**, e.g. asking for mean value over a dataset, is privacy preserving.
      * The mechanisms are designed based on the idea that **adding noise** to individual records or query results can mask out the privacy information locally while preserving an accurate aggregation over the dataset.
    * Categories
      * Sensitivity Mechanisms
      * Exponential Mechanisms

## Distributed ML (DML)

There are two main streams of DML: *Scalability-Motivated DML & Privacy-Motivated DML.* Both of the two paradigms adopt parallel computing technologies. So, the two concepts are not mutually exclusive. The key difference between the two families stems from their objectives. Informally, Scalability-Motivated DML **still collect data in data silos** and then apply distributed computing, which means the methods are not (necessarily) privacy-preserving.

* Scalability-Motivated DML
  * Motivations
    * Memory/Storage Shortage
    * Access to More Computational Resources
  * Technics
    * Data Parallelism
    * Model Partition
    * Graph-Centric Approach
    * Task-Centric Approach
    * Hybrid Parallelism
* Privacy-Motivated DML
  * Note
    * This note focus on Horizontal FL mainly
  * Motivations
    * Access to more (sensitive) training data
    * Privacy-Preservation
  * Technics
    * Server-Client Architecture
      * Procedure
        * Step 1: Local model training, and upload encrypted gradient/model to sever. The encryption can be performed with any method such as HE, DP, or SS .
        * Step 2: Server perform secured aggregation.
        * Step 3: Server dispatch update to clients.
        * Step 4: Client decryption and update model locally.
      * Characteristics
        * Assume a Honest-but-Curious Server
    * Peer-to-Peer Architecture
      * Procedure
        * Step 1: Local model training, and pass the updated model to next peer.
        * Step 2: Iterate until some stopping criteria has been met. There are two parameter passing schemas: *Cyclic Transfer* and *Random Transfer*
      * Characteristics
        * Central Parameter Server DO NOT Present
        * Gradient information of first peer was exposed (if no additional care was taken)
    * Federated Optimization
      * Key Properties
        * Non-IID Dataset
        * Unbalanced Dataset Size
        * Large Number of Diverse Participants
        * Unstable Connection
      * Algorithms (for Horizontal FL)
        * FedAvg (Gradient Avg & Model Avg)
        * Secured FedAvg
      * Improvements
        * Communication Efficiency
          * Sketched Updates
          * Structured Updates
            * Deep Gradient Compression (DGC)
        * Participants Selection
      * Challenges & Prospect
        * Federated Hyper-Parameter Tuning
        * Reward Mechanism 
          * To encourage more participants
        * Anti-Fraud
        * Training Process Control
          * E.g. Remote Early Stopping
        * Participant Management with Varying Reliability
          * E.g. Different connection stability