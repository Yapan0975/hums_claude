# RQ3 — Uncertainty-aware traversability

## Hypothesis (research_plan v3 §2 H3)

On SemanticSpray passive replay, an uncertainty-gated traversability head
produces **≥ +10 pp safe-region recall** AND **≥ −30 % false-traversable
rate** vs an R2-style hard-rule baseline, at equal coverage.

## Datasets

* SemanticSpray (D-e in research_plan v3 §4.1).

## Systems

* EvidLife-Map (full): vacuity-gated traversability head.
* EvidLife-Map (A-4 ablation off): hard-rule baseline.
* R2-reimpl: "unlabeled = untraversable".
* nvblox-bare: geometric-only height threshold.

Estimated compute: 1 GPU-day (research_plan v3 §4.5).
