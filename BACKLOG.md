# Football Tracker — Backlog

## 🟡 Parked / Ready to implement

### GK Manual Interval Override
**Context**: Some windows are misclassified not because of model error but because of real-world events (e.g. injury stoppage mid-session where the tracked player waited and moved slowly — correctly flagged as GK-like by speed, but actually neither role).

**Proposed solution**:
- Annotate time intervals with a reason (injury stoppage, break, drill change, warmup…)
- Auto-re-classify all windows falling inside the marked interval
- Show overridden windows visually distinct in all charts (greyed out / different marker)
- Store annotations in a `session_events.json` sidecar file alongside the CSV so they travel with the data
- Reason propagates to hover tooltips and the full window table
- Auditable — original label preserved alongside override label

**Status**: Design agreed, implementation not started. Return when prioritised.

---

## ✅ Stable

### GK Clustering Page — v1
All 6 sections working and stable:
1. 📍 2-D position scatter (rule-based + KMeans)
2. ⚡ Speed violin plots by role
3. 🕐 Role timeline
4. 🧊 3-D scatter (x, y, speed)
5. 🎯 Combined-score view (position vs mobility)
6. 🔬 PCA projection with loading arrows + table

**Known future improvement**: cluster stabilization (KMeans can flip GK/Outfield label between runs — consider majority vote across restarts, or seed-lock with rule-based validation check).
