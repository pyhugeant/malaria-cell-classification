# Mitosis Meta-to-Ana Pipeline (Fluorescence Microscopy)

A lightweight and reproducible pipeline for mitosis event localization and temporal estimation from fluorescence microscopy images.

---

## 📱 Project QR Code

Scan to open this repository on mobile:

<p align="center">
  <img src="https://api.qrserver.com/v1/create-qr-code/?size=200x200&data=https%3A%2F%2Fgithub.com%2Fpyhugeant%2Fmitosis-meta-ana" alt="QR Code" />
</p>

---

## 🔬 Overview

This project focuses on detecting mitotic events using a heatmap-based approach.

Main tasks:
1) Localize mitosis positions (meta / ana stages)
2) Estimate temporal duration:
   
   DeltaT = tana_start - tmeta_start

The pipeline is designed to be:
- Minimal
- Reproducible
- Easy to extend for research experiments

---

## ⚙️ Method Summary

- Input: fluorescence microscopy image sequences
- Model:
  - Encoder + U-Net style decoder
  - Heatmap regression (Gaussian targets)
- Output:
  - Peak detection -> mitosis coordinates
  - Optional temporal estimation

---

## 📁 Project Structure

