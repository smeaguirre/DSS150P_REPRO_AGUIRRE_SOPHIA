# Instructor Approval Request: Baseline Deviation

**Date:** October 13, 2026
**To:** DSS150P Instructor
**From:** Sophia Aguirre
**Subject:** Request for Approval - Historical Baseline Deviation (0.7145)

**Context:**
The historical prototype notebook reported an accuracy/metric of `0.7145`. Our engineered, fully deterministic pipeline yields a strict Multiclass Accuracy of `0.6249`.

**Investigation (Ablation Findings):**
1. **Scale Invariance:** XGBoost is a tree-based ensemble. Global scaling leakage (fitting standard scalers before splitting) does not mathematically inflate tree-based accuracy. 
2. **Duplicate Leakage:** Leaving all 24,000 cross-genre duplicates in the dataset only shifts accuracy to `~0.635`.
3. **Metric Mismatch:** Sweeping alternative metrics reveals that Regression R² on raw popularity yields `~0.17` and One-vs-Rest ROC-AUC yields `~0.74`. 

**Conclusion & Request:**
In accordance with the laboratory directive that *"a model that is slightly less accurate but fully reproducible is preferable to an undocumented model,"* we have isolated the classification pipeline to ensure 100% deterministic reproducibility rather than chasing the un-reproducible 0.7145 metric. I request official approval to accept the `0.6249` deterministic accuracy as the new verified baseline.

**Instructor Response:** 
*[Pending Instructor Signature/Approval]*
