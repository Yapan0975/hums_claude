# Project: EvidLife-Map (R2 Improvement Paper)

Target venue: **IROS 2027** (deadline ~2027-03)
Plan version: **v3** (research_plan.md)
Hardware: RTX 4060 (16 GB) + Jetson Orin NX
Baseline: R2 = Jiao et al. T-ASE 2024 (arXiv 2412.00291)

## Hard rules for this project

### Dual-format paper drafts (永久工作规则)

Every paper draft I write must be produced as **two formats in the same step**:

1. `drafts/paper_vN.md`   — Markdown source
2. `drafts/paper_vN.html` — Rendered HTML (TIP 精读 template style, MathJax, `<mark>` TBD highlights)

Optional: `drafts/_h5/paper_vN.html` if mobile preview requested.

Don't deliver just the `.md` and wait for user to ask "where's the HTML" — render both immediately. This applies to:
- Any paper version (v1 skeleton through camera-ready)
- Cover letters
- Response to reviewers (if formal)

NOT for: lit_scan / evidence_map / risk_log / R&R notes / integrity reports / code READMEs.

### Citations

Use `[bibkey]` form in drafts (`[schmid2024khronos]`). Format-convert to `[N]` numbers only at final-render stage. All citations must be in `refs/refs.bib` (60 entries). No citing from memory.

### Placeholders

- `XX.X` for numeric TBD (highlight `<mark>` yellow)
- `[TBD-RQx]` for paragraph-level TBD (highlight `<mark class="tbd">` red)
- `[NR]` for "Not Reported by original" — permanent
- `[TBD: impl]` for implementation details awaiting code

### Budget discipline

IEEE 8-page main + ≤30 refs. Current paper_v2 at 132% of outline budget — trim required at camera-ready.

### Risk gates

6 Go/No-Go criteria (G-1 through G-6) gate progression. See `code/risk_log/go_nogo_checklist.md`.
