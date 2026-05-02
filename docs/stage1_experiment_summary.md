# Stage 1 Experiment Summary

## Project
**MITOSIS-META-ANA**

## Document Purpose
This document summarizes the current first-stage experiment, including the model configuration, observed training behavior, interpretation of the results, and the recommended direction for the next round of ablation studies.

---

## 1. Experiment Objective
The original goal of the current experiment was to jointly learn:

1. **Meta localization** as the primary task
2. **Ana supervision** as an auxiliary task

The broader downstream goal is not to build a strong standalone ana detector. Instead, the real objective is:

- detect the **meta position** reliably,
- estimate **meta_start**,
- estimate **ana_start**, and
- compute the mitotic time interval:
  
  `mitotic duration = ana_start - meta_start`

Under this objective, the ana branch should be treated as an auxiliary signal only if it improves the final temporal estimation pipeline.

---

## 2. Current Experimental Setup

### 2.1 Model
- Model scale: **tiny**
- `base_channels = 64`
- Temporal module: **ConvLSTM**
- `hidden_channels = 128`

### 2.2 Multi-task Supervision
#### Meta head
- Target type: **Gaussian heatmap**
- Loss: **MSE**
- Weight: **20**
- Heatmap sigma: **6**

#### Ana head
- Target type: **binary disk**
- Loss: **BCE**
- Weight: **1**
- Disk radius: **30**

### 2.3 Training Setup
- Epochs: **50**
- Batch size: **2**

---

## 3. Observed Training Results

### 3.1 Meta loss
- Initial value: approximately **0.20**
- Rapid drop during the first few epochs
- Stabilized around **0.08** after approximately **epoch 5**

### 3.2 Ana loss
- Initial value: approximately **0.97**
- Very small decrease during the first few epochs
- Stabilized around **0.95** after approximately **epoch 5**

### 3.3 Overall pattern
Both branches plateaued very early, but the two heads behaved very differently:

- the **meta branch learned something useful**, although improvement stopped early,
- the **ana branch showed almost no meaningful learning**.

This early saturation strongly suggests that the current setup is not well aligned with the real task objective.

---

## 4. Interpretation

### 4.1 Meta is learnable, but the current setup reaches a shallow plateau
The meta loss dropped substantially from 0.20 to 0.08, which indicates that the model is able to learn meaningful meta-related spatial structure. However, the fact that this improvement largely stopped by epoch 5 suggests one or more of the following:

- the current optimization setting reaches a coarse local minimum too early,
- the temporal branch increases optimization difficulty,
- the multi-task formulation interferes with the primary task,
- the current heatmap sharpness may not be optimal for precise point localization.

### 4.2 Ana is currently not functioning as a useful auxiliary task
The ana loss changed only from 0.97 to 0.95, which is too small to interpret as meaningful learning. In practice, this means the ana branch is currently not contributing useful supervision.

Possible reasons include:

- the ana target definition is not well matched to real ana morphology,
- hard disk supervision is too coarse,
- BCE is not the most suitable loss for this target,
- the task is too difficult relative to the information content and model capacity,
- the auxiliary branch is consuming model capacity without helping the primary task.

### 4.3 The current task formulation may not reflect the real downstream objective
The downstream goal is not to segment or localize ana precisely in space. The downstream goal is to estimate **when ana starts**, relative to meta. Therefore, using a dense spatial ana target may be an indirect and inefficient formulation.

This suggests that the current ana branch may be solving the wrong problem.

---

## 5. Main Conclusion from Stage 1

### Conclusion 1
**Meta should remain the core task.**

The current experiment confirms that meta supervision is learnable and should remain the center of the model design.

### Conclusion 2
**The current ana branch should not be treated as an equal task.**

Given the almost flat ana loss curve, the current ana branch is not justified as a full parallel task under the present formulation.

### Conclusion 3
**The next priority is to build a strong meta-only baseline.**

Before tuning ana further, it is necessary to answer the following question:

> How well can the model perform if it is optimized only for meta localization?

This is the most important next step, because it will reveal whether the current ana branch is helping, neutral, or harmful.

### Conclusion 4
**Ana should be reconsidered as a weak auxiliary or temporal-state signal, not necessarily as a spatial detection head.**

If ana is retained in later experiments, it should most likely be reformulated as one of the following:

- a weak auxiliary branch with low loss weight,
- a softer target such as a soft disk or truncated Gaussian,
- or a temporal/state classifier for identifying `ana_start`.

---

## 6. Recommended Next Direction
The next stage should prioritize a clean ablation sequence centered on the primary objective.

### Immediate next step
Run a **meta-only baseline** with the ana branch removed.

### Follow-up priorities
1. optimize the meta branch first,
2. test whether ConvLSTM is truly beneficial,
3. reintroduce ana only as a weak auxiliary if justified,
4. consider replacing the spatial ana head with a temporal/state prediction head.

---

## 7. Practical Implication for the Project
At this point, the project should shift from a symmetric multi-task mindset to a **goal-driven design**:

- **Primary goal:** accurate meta localization and stable meta_start estimation
- **Secondary goal:** reliable estimation of ana_start only to support the final mitotic duration calculation

Therefore, future model decisions should be evaluated primarily by:

- meta localization accuracy,
- meta_start frame accuracy,
- ana_start frame accuracy,
- and the final error of `ana_start - meta_start`.

Training loss alone should not be used as the main decision criterion.

---

## 8. Summary Statement
The first-stage experiment shows that the current model can learn the meta task to a limited extent, but the current ana formulation is ineffective. Since the true project goal is to estimate mitotic timing rather than to build a strong ana spatial detector, the next experimental stage should focus on a stronger meta-only baseline and reintroduce ana only if it clearly improves the downstream temporal objective.
