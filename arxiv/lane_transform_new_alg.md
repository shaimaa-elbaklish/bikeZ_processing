Let me re-read the final algorithm carefully before summarising.Now I have a complete picture. Here's the full algorithm overview and the answer to your question.

---

## Algorithm overview — `tools_lane_coords_V4`

### One-time setup (`setup_registry`)
Before the vehicle loop: build spline LUTs, pre-expand validity polygons (`_poly_expanded`), pre-expand and pre-prepare intersection area polygons (`__intersection_area_*_expanded/prepared`).

---

### Per-trajectory: `assign_segments` — sequential chain loop

Iterates up to `max_chain_length` times. Each iteration consumes one fragment (remaining unmatched trajectory) and appends one matched segment to the chain.

**Step 1 — Polygon walk**
For every candidate segment: walk the fragment from index 0, find the first continuous run inside `poly.buffer(tolerance)` → `(entry_idx, exit_idx)`.

Group assignment:
- **Group A**: `entry_idx == 0`
  - Lane segments additionally checked: if `fragment_xy[0]` is inside any `intersection_area_*` → demoted to Group B (turns should claim intersection-starting trajectories)
  - Turn segments: always Group A if `entry_idx == 0`
- **Group B**: `entry_idx > 0`
- **Group C**: no polygon match → rejected

**Step 2 — Choose scoring group**
- Group A non-empty → score Group A, `is_fallback=False`
- Group A empty, Group B non-empty → score Group B, `is_fallback=True`
- Both empty:
  - **Iteration 0 only**: expand to all segments and retry
  - **Iteration > 0**: stop chain (cyclist has exited scene)

**Step 3 — Score each candidate**
On its `[entry_idx : exit_idx]` window:
1. Detect `is_reverse` from net s progression sign
2. Veto reversed turns (`score=inf`)
3. For lane segments: exclude points inside any `intersection_area_*` from scoring
4. Hard lateral veto: `median_d` outside `[-(d_veto_right+1), +(d_veto_left+1)]` — with d_left/d_right swapped for reverse
5. Score = `W_DIST·dist + W_HEAD·head + W_ARC·arc + W_PILE·pile(turns only) + W_REVERSE(reverse lanes)`

**Step 4 — Accept / reject best**
- `score > FORCED_MATCH_THRESHOLD` → stop chain
- `score > POOR_MATCH_THRESHOLD` → `match_quality='poor'`
- else → `match_quality='good'`
- `is_fallback=True` → `match_quality='fallback'`

**Step 5 — Handoff at s_change**
Project polygon window onto spline → s_win. Find first s crossing in travel direction:
1. Check primary `s_change` (main intersection) → returns `(idx, 's_change')`
2. If not reached, check extra `s_change_*` keys (secondary junctions) → returns `(idx, key)`
3. Nothing crossed → `(exit_idx, None)`

Special cases: turns and departure segments return `(len(s_win), None)`.

**T-junction confirmation** (only when `s_change_key_fired` is a secondary key):
- Take remaining fragment after handoff
- Find minor road segments (different geometry_key, departure of matching movements)
- Check if any remaining point is inside minor road polygon AND past `s_change` on minor road
- **Confirmed** → turn happened, proceed with handoff and MattInt candidates
- **Not confirmed** → straight through:
  - Re-run primary `s_change` crossing only
  - `handoff_local` = either primary crossing index or `exit_idx`
  - `s_change_key_fired` = `'s_change'` or `None`

**Step 6 — Role and next candidates**
Role assigned by chain position via `_assign_role` (iteration 0 uses `_infer_role_from_registry`).

`get_next_candidates` filtered by `s_change_key_fired`:
- `'s_change'` → only MainInt turn candidates
- `'s_zollstr_west_yield'` → only MattInt turn candidates for that direction
- `None` → no filtering

For reverse traversal: looks up via opposite-direction key + flipped role.

Chain extension: departure segments also searched as approach (handles through-roads serving multiple intersections).

---

## On intersection type checking

**No — we don't explicitly check intersection type** (`'standard'` vs `'T-junction'`). The distinction is implicit:

The T-junction confirmation only triggers when `s_change_key_fired` is a **secondary boundary key** (not `'s_change'`). Secondary keys only exist on through-roads at T-junctions — they're defined via `extra_changes` in the site file. Standard intersections only have the primary `s_change`, so they never trigger the confirmation logic.

In other words: the presence of `extra_changes` on a geometry axis is what implicitly marks it as a T-junction through-road. No explicit `intersection_type` flag is needed — the registry structure encodes it naturally.

The only case where an explicit type flag would add value is if you wanted to display it in the folium map or use it in fleet-level analysis. For the matching algorithm itself, it's not needed.