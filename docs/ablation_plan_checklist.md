# Ablation Plan Checklist

## Project
**MITOSIS-META-ANA**

## Purpose
This checklist defines the prioritized ablation plan for the next experimental stage. The plan is organized around the true downstream objective:

- robust **meta localization**,
- accurate **meta_start** estimation,
- accurate **ana_start** estimation,
- and improved estimation of the mitotic interval:
  
  `ana_start - meta_start`

The key principle is:

> Build the strongest possible meta baseline first, then test whether ana provides real downstream value.

---

## Global Rules for All Experiments

### Fixed settings unless explicitly changed
- [ ] Keep the same train/validation split
- [ ] Keep the same data preprocessing
- [ ] Keep the same augmentation policy
- [ ] Keep the same optimizer
- [ ] Keep the same evaluation script
- [ ] Keep the same postprocessing logic
- [ ] Keep random seeds fixed when possible

### Core metrics to record for every experiment
- [ ] Meta localization error
- [ ] Meta detection recall at fixed distance thresholds
- [ ] Meta_start frame error
- [ ] Ana_start frame error if applicable
- [ ] Final mitotic interval error: `|(ana_start - meta_start)_pred - (ana_start - meta_start)_gt|`
- [ ] Training stability
- [ ] Validation trend consistency

### Suggested logging items
- [ ] Experiment ID
- [ ] Date
- [ ] Model configuration
- [ ] Loss configuration
- [ ] Best validation epoch
- [ ] Qualitative observations
- [ ] Keep / reject decision

---

# Experiment 1 - Meta-Only Baseline

## Priority
**Highest**

## Objective
Determine how well the model performs when optimized only for the primary task.

## Hypothesis
The current ana branch may be interfering with the main task. Removing it may improve meta learning.

## Changes
- [ ] Remove the ana head
- [ ] Keep only the meta head
- [ ] Keep meta target as Gaussian heatmap
- [ ] Keep `sigma = 6`
- [ ] Keep meta loss as MSE
- [ ] Keep tiny model with `base_channels = 64`
- [ ] Keep ConvLSTM for the first baseline run
- [ ] Keep `hidden_channels = 128` for the first baseline run

## Training suggestion
- [ ] Train for 100-150 epochs
- [ ] Keep batch size at 2
- [ ] Reduce learning rate if early plateau persists

## What to compare against
- [ ] Current multi-task model

## Success criteria
- [ ] Lower meta localization error
- [ ] Better meta_start frame accuracy
- [ ] More stable validation behavior

## Decision rule
- [ ] If meta-only is better, ana should no longer be treated as a full parallel task
- [ ] If meta-only is not worse, ana is not currently justified

---

# Experiment 2 - Meta Heatmap Sigma Ablation

## Priority
**Second**

## Objective
Find the best heatmap sharpness for meta localization.

## Hypothesis
`Sigma = 6` may be too broad for precise point localization.

## Variants
- [ ] Exp2-a: `sigma = 4`
- [ ] Exp2-b: `sigma = 5`
- [ ] Exp2-c: `sigma = 6` as control

## Fixed settings
- [ ] Use the best structure from Experiment 1
- [ ] Keep all other training settings unchanged

## What to examine
- [ ] Peak sharpness of predicted heatmaps
- [ ] Meta point localization accuracy
- [ ] Meta_start frame accuracy
- [ ] Validation robustness

## Decision rule
- [ ] Select the sigma with the best validation localization and timing performance
- [ ] Use the winning sigma in all later experiments

---

# Experiment 3 - ConvLSTM Ablation

## Priority
**Third**

## Objective
Test whether the temporal module is actually helping the primary task.

## Hypothesis
ConvLSTM may be too heavy for the current data scale or may complicate optimization.

## Variants
- [ ] Exp3-a: no ConvLSTM
- [ ] Exp3-b: ConvLSTM with `hidden_channels = 64`
- [ ] Exp3-c: ConvLSTM with `hidden_channels = 128` as control

## Fixed settings
- [ ] Use the best sigma from Experiment 2
- [ ] Keep all other settings unchanged

## What to examine
- [ ] Meta localization error
- [ ] Meta_start frame accuracy
- [ ] Convergence speed
- [ ] Early plateau behavior
- [ ] Validation stability

## Decision rule
- [ ] If no ConvLSTM performs best, remove the temporal module for the next stage
- [ ] If `hidden = 64` performs best, keep a lighter temporal model
- [ ] If `hidden = 128` performs best, retain the current temporal design

---

# Experiment 4 - Weak Auxiliary Ana Reintroduction

## Priority
**Fourth**

## Objective
Test whether ana can help as a weak auxiliary task after the meta representation is already stabilized.

## Hypothesis
Ana may be helpful only when introduced later and with a small weight.

## Training strategy
- [ ] First stage: train only meta for the initial 20 epochs
- [ ] Second stage: reintroduce ana with a small weight

## Variants
- [ ] Exp4-a: `lambda = 0.05`
- [ ] Exp4-b: `lambda = 0.10`
- [ ] Exp4-c: `lambda = 0.20`

## Fixed settings
- [ ] Use the best meta configuration from Experiments 1-3
- [ ] Keep the current ana target temporarily for this experiment

## What to examine
- [ ] Whether meta performance degrades
- [ ] Whether final mitotic interval estimation improves
- [ ] Whether ana shows any real learning behavior

## Decision rule
- [ ] If meta degrades, reject the ana auxiliary branch in its current form
- [ ] If final interval estimation improves without hurting meta, keep weak ana supervision
- [ ] If there is no meaningful gain, do not prioritize ana spatial supervision further

---

# Experiment 5 - Ana Target Reformulation

## Priority
**Fifth**

## Objective
Determine whether the current ana failure is mainly caused by the target definition.

## Hypothesis
A hard binary disk with BCE is too coarse and not well matched to ana morphology.

## Variants
- [ ] Exp5-a: hard disk + BCE
- [ ] Exp5-b: soft disk + BCE
- [ ] Exp5-c: truncated Gaussian + MSE or equivalent heatmap-style loss

## Fixed settings
- [ ] Use the best meta backbone from Experiments 1-3
- [ ] Use the best ana weight strategy from Experiment 4

## Optional radius sweep if needed
- [ ] Test `r = 12`
- [ ] Test `r = 16`
- [ ] Test `r = 20`
- [ ] Keep `r = 30` only as reference

## What to examine
- [ ] Whether ana loss finally decreases meaningfully
- [ ] Whether predicted ana activation becomes more localized and interpretable
- [ ] Whether the final interval error improves

## Decision rule
- [ ] If soft targets outperform hard disk targets, stop using hard disk supervision
- [ ] If target reformulation still does not help downstream timing, deprioritize ana spatial learning

---

# Experiment 6 - Replace Ana Spatial Head with a Temporal or State Head

## Priority
**Sixth**

## Objective
Reformulate ana supervision to match the real downstream goal.

## Hypothesis
Ana is fundamentally a temporal transition problem rather than a spatial localization problem.

## Candidate formulations
- [ ] Exp6-a: binary frame-level classifier for `is_ana`
- [ ] Exp6-b: three-class state classifier: `pre-meta / meta / ana`
- [ ] Exp6-c: transition classifier for detecting the onset of ana

## Fixed settings
- [ ] Keep the best meta localization configuration from previous experiments
- [ ] Use the same evaluation protocol for downstream timing comparison

## What to examine
- [ ] Ana_start frame accuracy
- [ ] Final mitotic interval error
- [ ] Stability of temporal predictions
- [ ] Whether this formulation is more interpretable than the ana spatial head

## Decision rule
- [ ] If a temporal/state head gives better timing estimates, replace the ana spatial head in the main pipeline
- [ ] If performance is similar but the temporal head is simpler and more interpretable, prefer the temporal formulation

---

# Recommended Execution Order

## Round 1 - Strengthen the primary task first
- [ ] Experiment 1: Meta-only baseline
- [ ] Experiment 2: Meta sigma ablation
- [ ] Experiment 3: ConvLSTM ablation

## Round 2 - Reintroduce ana only if justified
- [ ] Experiment 4: Weak auxiliary ana
- [ ] Experiment 5: Ana target reformulation

## Round 3 - Align the task with the real objective
- [ ] Experiment 6: Temporal or state-based ana modeling

---

# Final Decision Framework

Use the following questions after each experiment:

- [ ] Does this change improve meta localization?
- [ ] Does this change improve meta_start estimation?
- [ ] Does this change improve ana_start estimation?
- [ ] Does this change reduce the final interval error?
- [ ] Is the added complexity justified by downstream gain?

If the answer is no for the final interval objective, the change should not be prioritized further.

---

# Short Summary

The ablation sequence is designed to answer the following in order:

1. Can meta perform better by itself?
2. What is the best supervision sharpness for meta?
3. Is ConvLSTM truly necessary?
4. Can ana help as a weak auxiliary?
5. Is the current ana failure caused by the target design?
6. Should ana be reformulated as a temporal/state prediction task?

This order minimizes wasted effort and keeps the project aligned with the actual biological timing objective.
