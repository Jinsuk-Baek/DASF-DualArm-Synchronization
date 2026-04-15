# Dual-Arm Synchronization Framework (DASF)

This repository contains the implementation of synchronization methods evaluated in the paper on dual-arm robotic coordination under timing uncertainty.

## Overview

The repository provides five execution models with increasing levels of coordination:

- **Sequential (No-sync):** No explicit synchronization between manipulators  
- **Timestamp-based:** Time-driven synchronization using a shared global timestamp  
- **Event-based:** Event-driven synchronization using thread signaling  
- **TLS:** Task-Level Synchronization using barrier-based coordination  
- **TLS+MLS:** Proposed method combining task-level synchronization and motion-level adaptive compensation  

Each script executes the same multi-phase task sequence and logs synchronization performance metrics.

---

## Hardware and Execution Environment

All experiments were conducted on a **CPU-only control PC** without GPU acceleration.  
The robotic platform consists of **two uFactory Lite6 collaborative manipulators**.

The proposed synchronization framework operates at the software control layer and introduces negligible computational overhead, enabling real-time execution without specialized hardware support.

---

## Repository Structure

Each Python script corresponds to a specific synchronization method:

- `DualArms_4Phase_TaskLike_Sequential_NoSync_CSV.py`
- `DualArms_4Phase_TaskLike_Timestamp_Based_CSV.py`
- `DualArms_4Phase_TaskLike_Event.py`
- `DualArms_4Phase_TaskLike_TLS.py`
- `DualArms_4Phase_TaskLike_TLS_MLS.py`

---

## Execution

Each script can be executed independently.

Example:

```bash
python DualArms_4Phase_TaskLike_TLS_MLS.py
