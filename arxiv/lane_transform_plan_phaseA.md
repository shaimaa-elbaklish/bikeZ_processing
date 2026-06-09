The three layers were:

geometry_store — raw geometric objects, one per physical road. Stores the spline, positive_dir, total_length, line_wgs84, and now also s_stop and s_yield in spline-native arc-length.
segment_registry — directed, typed segments. One entry per traversable unit (e.g. Roentgenstr_EB, Roentgenstr_WB). References geometry_store by geometry_key, adds mode, bike_lane info, and the direction-aware logic for approach/departure using s_stop/s_yield from geometry_store.
movement_registry — named sequences. One entry per observable trajectory type through the intersection (e.g. Roentgenstr_EB_2_LangstrN_NB). Just an ordered list of segment_registry keys: [approach_key, turn_key, departure_key].


# Lane Coordinate Transform — Output Column Reference

Output of `to_lane_coordinates()` for one vehicle trajectory.  
All lane coordinate columns are `NaN` for unmatched rows.

---

## Matching columns

| Column | Type | Description |
|--------|------|-------------|
| `movement_key` | `str` | Named movement the vehicle was assigned to, e.g. `Roentgenstr_EB_2_LangstrN_NB`. Consistent across all rows belonging to the same chain. `NaN` if unmatched. |
| `segment_id` | `str` | Key of the matched segment within the movement, e.g. `Roentgenstr_EB` or `turn_Roentgenstr_EB_2_LangstrN_NB`. `NaN` for unmatched rows. |
| `segment_type` | `str` | `'lane'` or `'turn'`. |
| `segment_role` | `str` | Role within the movement: `'approach'`, `'turn'`, or `'departure'`. |

---

## Lane coordinates

The coordinate system resets to `s = 0` at each segment entry (approach start or turn start). `d` is defined relative to the segment centerline in the direction of travel.

| Column | Unit | Description |
|--------|------|-------------|
| `s` | m | Arc-length along the segment centerline in the **travel direction**, always starting at 0 at segment entry. Increases monotonically under normal forward travel. For approach segments: 0 at the far end, `s_stop` at the stop-line. For departure segments: 0 at the yield-line, increasing toward the far end. For turns: 0 at the entry, `L_turn` at the exit. |
| `d` | m | Signed lateral offset from the centerline. **Positive = left of travel direction**, negative = right. A cyclist in the right-hand bike lane has negative `d` on a shared road. |

---

## Speeds

Decomposed from EKF speed (`speed_ekf`) and heading (`angle_ekf`) using the spline tangent and normal vectors at the projected point.

| Column | Unit | Description |
|--------|------|-------------|
| `s_dot` | m/s | Longitudinal speed — component of velocity along the centerline tangent. Positive = forward travel. Negative values indicate slowing/reversing (see `s_decreasing`). |
| `d_dot` | m/s | Lateral speed — component of velocity along the centerline normal. Positive = moving left, negative = moving right. |

---

## Accelerations

Decomposed from EKF scalar acceleration (`a`) and heading (`angle_ekf`), projected onto the same tangent/normal frame as the speeds.

| Column | Unit | Description |
|--------|------|-------------|
| `s_ddot` | m/s² | Longitudinal acceleration along the centerline. Positive = accelerating forward, negative = braking. |
| `d_ddot` | m/s² | Lateral acceleration perpendicular to the centerline. Non-zero during lane changes or cornering. |

---

## Flags and diagnostics

| Column | Type | Description |
|--------|------|-------------|
| `s_decreasing` | `bool` | `True` when `s_dot < -0.5 m/s` — the vehicle is moving backwards along the segment (U-turn detection, stopped-and-reversing). |
| `in_bike_lane` | `bool` / `NaN` | `True` if the vehicle's lateral position `d` falls within the dedicated bike lane boundary (± `BIKE_LANE_TOLERANCE = 0.2 m`). `NaN` if the segment has no bike lane geometry or if `s` falls outside the range where bike lane boundaries are defined. |
| `d_to_bike_boundary` | m / `NaN` | Signed distance from the vehicle to the near edge of the bike lane boundary, in native spline coordinates. Negative = vehicle is inside the bike lane (closer to centerline than the boundary), positive = outside. `NaN` same conditions as `in_bike_lane`. |

---

## Sign conventions summary

```
Travel direction →

Centerline  ─────────────────────────────────────────────►
                         d > 0 (left)
            ─────────────────────────────────────────────►
                         d < 0 (right)  ← bike lane side (Swiss right-hand traffic)
```

```
s increases in travel direction (always from 0 at segment entry)
s_dot > 0  →  moving forward
s_dot < 0  →  moving backward  (s_decreasing = True)
d_dot > 0  →  drifting left
d_dot < 0  →  drifting right
```

---

## NaN conditions

A row has `NaN` in all lane coordinate columns when:

- The vehicle was not matched to any segment (e.g. entirely on an unregistered road such as Mattengasse)
- The row falls in the unregistered prefix before the first registered segment entry
- The row falls between two matched segments (gap at handoff — rare)

`in_bike_lane` and `d_to_bike_boundary` additionally return `NaN` when:

- The segment has no dedicated bike lane (`bike_lane = None` in `segment_registry`)
- The projected `s` value falls outside the `s_domain` where bike lane boundaries were drawn in swisstopo

---

## Coordinate system reference

All spline computations are performed in **local EPSG:2056** coordinates:

```python
x_local = x_LV95 - geometry_store['x_offset']
y_local = y_LV95 - geometry_store['y_offset']
```

Input columns used: `x_ekf`, `y_ekf` (already in local coords), `speed_ekf`, `angle_ekf`, `a`.