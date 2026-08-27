# Dual-Arm Synchronization Framework (DASF)

This repository contains the source code used for the experimental evaluation of the **Dual-Arm Synchronization Framework (DASF)** presented in the following article:

**Jinsuk Baek, Sanhong Lee, and Ju Hong Park**  
**"Dual-Arm Synchronization Framework for Smart Kitchen Robotic Systems"**  
*IEEE Access*, 2026.

The DOI and complete bibliographic information will be added after final publication.

---

## Overview

DASF is a two-layer synchronization framework for dual-arm robotic systems operating under timing uncertainty. It consists of two complementary synchronization mechanisms:

- **Task-Level Synchronization (TLS):** barrier-based phase-level coordination that aligns the execution of the two manipulators at task-phase boundaries.
- **Real-Time Motion-Level Synchronization (RT-MLS):** execution-level compensation that monitors timing differences during motion and applies adaptive proportional speed scaling.

The repository provides the experimental implementations used to evaluate DASF on a real dual-arm robotic platform.

The experiments include four cooking-inspired manipulation scenarios, Cartesian tracking validation, and additional robustness evaluations under asymmetric workload and intermittent execution disturbances.

---

## Hardware and Execution Environment

The experiments were conducted using:

- Two **uFactory Lite6** collaborative manipulators
- A centralized control PC
- Wired Ethernet communication through an Ethernet switch
- Python-based robot control using the **uFactory xArm Python SDK**
- CPU-only execution without GPU acceleration

The control PC independently interfaces with both manipulators and coordinates their execution through the proposed TLS and RT-MLS mechanisms.

The main experimental parameters are:

- Control frequency: **100 Hz (10 ms control cycle)**
- TLS synchronization tolerance: **5 ms**
- RT-MLS proportional gain: **0.003**
- RT-MLS monitoring interval: **50 ms**
- RT-MLS deadband: **3 ms**
- Speed-scaling bounds: **[0.95, 1.05]**

---

## Repository Structure

The source code is organized according to the experimental evaluations reported in the paper.

```text
DASF-DualArm-Synchronization/
│
├── Ingredient_Transfer/
│   └── Source code for the Ingredient Transfer experiments
│
├── Chopping/
│   └── Source code for the Chopping experiments
│
├── Kneading/
│   └── Source code for the Kneading experiments
│
├── Scooping/
│   └── Source code for the Scooping experiments
│
├── Cartesian/
│   └── Source code for the Cartesian tracking accuracy experiments
│
├── AsymWorkload/
│   └── Source code for the asymmetric workload robustness experiment
│
├── Disturbance/
│   └── Source code for the intermittent execution disturbance experiment
│
└── README.md
```

---

## Main Synchronization Experiments

The four task folders correspond to the cooking-inspired manipulation scenarios evaluated in the paper.

### Ingredient Transfer

A multi-phase coordinated manipulation task including approach, lift, transfer, placement, retreat, and return motions.

### Chopping

A repetitive vertical-motion task representing synchronized chopping or cutting actions.

### Kneading

A planar coordinated-motion task involving forward, lateral, and diagonal pressing motions.

### Scooping

A compound multi-axis trajectory involving reach, downward motion, lifting, and forward movement.

The source code in these folders implements the synchronization methods used for the main experimental comparison, including TLS and the proposed TLS+RT-MLS configuration.

The implementations record synchronization metrics including:

- Dispatch gap between the two manipulators
- Finish gap between the two manipulators
- Phase-level execution timing
- RT-MLS speed-scaling behavior, where applicable

These experiments correspond to the main synchronization evaluations reported in **Tables III and IV** of the paper.

---

## Cartesian Tracking Experiments

The `Cartesian/` folder contains the source code used for the additional Cartesian tracking validation reported in **Table V** of the paper.

For each synchronization checkpoint, the Cartesian position of the **tool center point (TCP)** is measured and compared with the predefined target position. Tracking error is calculated independently for the two manipulators because they employ different end-effectors.

The Cartesian tracking error is calculated as the Euclidean distance between the predefined target position and the measured TCP position.

These experiments evaluate whether the execution-level speed compensation introduced by RT-MLS preserves end-effector positioning accuracy while improving temporal synchronization.

---

## Robustness Experiments

Two additional experimental configurations are provided to evaluate DASF under more challenging execution conditions.

### Asymmetric Workload

The `AsymWorkload/` folder contains the implementation used to introduce a controlled asymmetric workload between the two manipulators.

This experiment evaluates whether RT-MLS can compensate for persistent execution-time differences while preserving the phase-level synchronization established by TLS.

### Intermittent Execution Disturbance

The `Disturbance/` folder contains the implementation used to evaluate synchronization performance under intermittent random execution disturbances.

The disturbances introduce temporary execution-time variations during coordinated operation, allowing the robustness of RT-MLS to be evaluated under non-nominal execution conditions.

These robustness experiments employ the same TLS and RT-MLS algorithms used in the main experiments without introducing additional online synchronization mechanisms.

---

## Execution

Each experimental script can be executed independently from its corresponding directory.

For example:

```bash
python <script_name>.py
```

Before execution, users should configure the network addresses of the two robotic manipulators according to their local experimental setup.

The required Python environment should include the **uFactory xArm Python SDK** and the standard Python packages imported by the corresponding experimental scripts.

Robot-specific parameters, including initial poses, motion coordinates, and safety settings, should be verified before executing the code on a physical robotic platform.

---

## Experimental Data

This repository provides the **source code used to conduct the experiments reported in the paper**.

CSV files generated during the physical robot experiments are not included in the repository. The experimental scripts generate the corresponding timing, synchronization, and tracking measurements during execution on the dual-arm robotic platform.

Because execution timing depends on the host operating system, robot controllers, communication latency, hardware configuration, and system scheduling, exact numerical results may vary across experimental platforms.

---

## Reproducibility

The repository is intended to provide the implementation details necessary to reproduce the synchronization mechanisms and experimental procedures described in the paper.

The main synchronization experiments reproduce the experimental configurations associated with **Tables III and IV**, while the additional source code supports the Cartesian tracking and robustness evaluations reported in the paper.

Users reproducing the experiments on different robotic platforms may need to modify:

- Robot IP addresses
- Robot initialization procedures
- Initial and target poses
- Motion coordinates
- End-effector configurations
- Safety parameters

The synchronization logic of TLS and RT-MLS can be retained independently of these platform-specific settings.

---

## Code Availability

The source code associated with this study is publicly available in this repository:

**DASF-DualArm-Synchronization**

The repository provides the implementations of TLS and RT-MLS together with the experimental programs used for the cooking-inspired task scenarios, Cartesian tracking validation, and robustness evaluations.

---

## Citation

If you use this code or the DASF framework in your research, please cite:

**J. Baek, S. Lee, and J. H. Park,  
"Dual-Arm Synchronization Framework for Smart Kitchen Robotic Systems,"  
IEEE Access, 2026.**

The complete volume, page numbers, and DOI will be added after final publication.

---

## Authors

**Jinsuk Baek**  
Department of Computer Science  
Winston-Salem State University  
Winston-Salem, NC, USA

**Sanhong Lee**  
LSB  
Gumi, Republic of Korea

**Ju Hong Park**  
Department of Convergence IT Engineering  
Pohang University of Science and Technology (POSTECH)  
Pohang, Republic of Korea

---

## License

Copyright © 2026 Jinsuk Baek, Sanhong Lee, and Ju Hong Park.  
All rights reserved.

This source code is provided for academic research and reproducibility
purposes. No warranty is provided regarding the operation, safety, or
suitability of the software for any particular application. Users are
responsible for verifying all robot-specific configurations and safety
parameters before execution on physical robotic systems.

The copyright and licensing terms of the associated IEEE Access article
are governed separately by the applicable publication agreement.
