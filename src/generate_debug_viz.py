"""
TITLE OF PAPAER
-------------------------------------------
Authors:        Shaimaa El-Baklish
Organization:   ETH Zürich, Switzerland, IVT - Institute for Transportation Planning and Systems
Development:    2025-2026
Submitted to:   JOURNAL
-------------------------------------------


Interactive HTML debug visualisation for one bicycle trajectory after
lane coordinate transform.

Usage
-----
    from generate_debug_viz import generate_bikelane_debug_map
    generate_bikelane_debug_map(
        bike_df_veh24,
        segment_registry,
        geometry_store,
        output_path='debug_bikelane_map.html'
    )
"""

import json
import math
import numpy as np

from tools_utils import _PROJ_2056_TO_LONLAT
from tools_utils import _local_to_latlon, _spline_xy_to_latlon
from tools_utils import _spline_xy_variable_offset_to_latlon
from tools_utils import w_bike_at, w_bike_label

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _safe(val):
    try:
        if math.isnan(val) or math.isinf(val):
            return None
    except (TypeError, ValueError):
        pass
    return val


def _build_traj_json(df):
    """Serialise trajectory to JSON."""
    lon_arr, lat_arr = _PROJ_2056_TO_LONLAT.transform(
        df["x_act_ekf"].values, df["y_act_ekf"].values
    )
    records = []
    for i, row in enumerate(df.itertuples(index=False)):
        records.append({
            "t":           float(row.time),
            "lat":         float(lat_arr[i]),
            "lon":         float(lon_arr[i]),
            "s":           _safe(row.s),
            "d":           _safe(row.d),
            "s_native":    _safe(row.s_native),
            "d_native":    _safe(row.d_native),
            "s_dot":       _safe(row.s_dot),
            "d_dot":       _safe(row.d_dot),
            "speed":       _safe(row.speed_ekf),
            "seg_id":      str(row.segment_id)   if row.segment_id   is not None else None,
            "seg_role":    str(row.segment_role) if row.segment_role is not None else None,
            "is_reverse":  bool(row.is_reverse)  if row.is_reverse   is not None else False,
            "in_bike_lane": int(row.in_bike_lane) if (
                row.in_bike_lane is not None
                and not (isinstance(row.in_bike_lane, float) and math.isnan(row.in_bike_lane))
            ) else 0,
        })
    return json.dumps(records)


def _chain_seg_keys(bike_df):
    """Ordered unique segment keys from bike_df in appearance order."""
    seen = []
    for val in bike_df["segment_id"]:
        if val is not None and not (isinstance(val, float) and math.isnan(float(val)
                                    if isinstance(val, float) else 0)):
            s = str(val)
            if s not in seen and s != 'nan' and s != 'None':
                seen.append(s)
    return seen


def _build_map_layers(chain_seg_keys, segment_registry, geometry_store,
                      n_pts=300):
    """
    For every segment in the chain build:
      - centerline polyline
      - validity polygon ring
      - change-point marker (lat/lon) from geometry_store s_change
      - optional bike lane band

    Returns a list of dicts ready for JSON embedding.
    """
    x_offset = geometry_store['x_offset']
    y_offset = geometry_store['y_offset']
    layers = []

    for seg_key in chain_seg_keys:
        if seg_key not in segment_registry:
            continue
        entry    = segment_registry[seg_key]
        geom_key = entry['geometry_key']
        geom     = geometry_store[geom_key]
        tck, unew, cum_dist = geom['spline']
        total_len = geom['total_length']

        # ── Centerline ─────────────────────────────────────────────────────
        s_cl  = np.linspace(0, total_len, n_pts)
        cl_ll = _spline_xy_variable_offset_to_latlon(
            tck, unew, cum_dist, s_cl, np.zeros(n_pts), x_offset, y_offset)

        # ── Validity polygon ────────────────────────────────────────────────
        vp = entry.get('validity_polygon')
        vp_ll = ([] if (vp is None or vp.is_empty)
                 else _local_to_latlon(*vp.exterior.coords.xy, x_offset, y_offset))


        # ── Change-point marker ─────────────────────────────────────────────
        # Use the approach_s_change_key if present (turn), else 's_change'
        s_change_key = entry.get('approach_s_change_key', 's_change')
        s_chg = geom.get(s_change_key) if s_change_key else geom.get('s_change')
        chg_ll = None
        if s_chg is not None:
            chg_ll = _spline_xy_to_latlon(
                tck, unew, cum_dist, float(s_chg), float(s_chg), 
                x_offset, y_offset, n=1)[0]

        layer = {
            "seg_key":    seg_key,
            "seg_type":   entry["type"],
            "centerline": cl_ll,
            "vp_ring":    vp_ll,
            "chg_pt":     chg_ll,
            "bike_band":  None,
            "bike_bnd":   None,
            "bike_far":   None,
            "w_bike":     None,
            "side":       None,
        }

        # ── Bike lane band ──────────────────────────────────────────────────
        bike_lane = entry.get("bike_lane")
        if (entry["type"] == "lane" and bike_lane is not None
                and "d_boundary_spline" in bike_lane):
            d_bnd_spl    = bike_lane["d_boundary_spline"]
            # w_bike       = bike_lane["w_bike"]
            side         = bike_lane["side"]
            s_min, s_max = bike_lane["s_domain"]

            s_bl  = np.linspace(s_min, s_max, n_pts)
            d_bnd = d_bnd_spl(s_bl)
            d_far = d_bnd + side * w_bike_at(bike_lane, s_bl)

            bnd_ll = _spline_xy_variable_offset_to_latlon(
                tck, unew, cum_dist, s_bl, d_bnd, x_offset, y_offset)
            far_ll = _spline_xy_variable_offset_to_latlon(
                tck, unew, cum_dist, s_bl, d_far, x_offset, y_offset)

            layer["bike_bnd"]  = bnd_ll
            layer["bike_far"]  = far_ll
            layer["bike_band"] = bnd_ll + far_ll[::-1] + [bnd_ll[0]]
            layer["w_bike"]    = w_bike_label(bike_lane)
            layer["side"]      = int(side)

        layers.append(layer)

    return layers


def _build_vrect_shapes(traj_records, time_key='t'):
    """
    Build Plotly shapes (vrects) for is_reverse and in_bike_lane flags.
    Returns {'reverse': [...shapes...], 'bike_lane': [...shapes...]}.
    Each shape is a dict ready for Plotly layout.shapes.
    """
    def _runs(records, flag_fn):
        """Extract contiguous runs where flag_fn(rec) is True."""
        runs = []
        in_run = False
        t0 = None
        for rec in records:
            if flag_fn(rec):
                if not in_run:
                    t0 = rec[time_key]
                    in_run = True
                t1 = rec[time_key]
            else:
                if in_run:
                    runs.append((t0, t1))
                    in_run = False
        if in_run:
            runs.append((t0, t1))
        return runs

    reverse_runs   = _runs(traj_records, lambda r: r.get('is_reverse', False))
    bike_lane_runs = _runs(traj_records, lambda r: r.get('in_bike_lane', 0) == 1)

    def _shapes(runs, color, opacity):
        return [{"type": "rect",
                 "xref": "x", "yref": "paper",
                 "x0": t0, "x1": t1,
                 "y0": 0,  "y1": 1,
                 "fillcolor": color,
                 "opacity": opacity,
                 "line": {"width": 0},
                 "layer": "below"} for t0, t1 in runs]

    return {
        "reverse":   _shapes(reverse_runs,   "#ef4444", 0.10),
        "bike_lane": _shapes(bike_lane_runs,  "#22c55e", 0.10),
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def generate_bikelane_debug_map(bike_df, segment_registry, geometry_store,
                                 output_path='debug_bikelane_map.html',
                                 n_spline_pts=300):
    """
    Parameters
    ----------
    bike_df          : DataFrame after to_lane_coordinates(), single vehicle
    segment_registry : dict
    geometry_store   : dict
    output_path      : path to write HTML
    n_spline_pts     : spline sampling resolution
    """
    veh_id = int(bike_df['veh_id'].iloc[0]) if 'veh_id' in bike_df.columns else '?'

    chain_keys  = _chain_seg_keys(bike_df)
    map_layers  = _build_map_layers(chain_keys, segment_registry,
                                     geometry_store, n_pts=n_spline_pts)
    traj_json   = _build_traj_json(bike_df)
    layers_json = json.dumps(map_layers)

    # vrect shapes (computed in Python so JS only needs to embed JSON)
    import json as _json
    traj_records = _json.loads(traj_json)
    vrects       = _build_vrect_shapes(traj_records)
    vrects_json  = json.dumps(vrects)

    lon_all, lat_all = _PROJ_2056_TO_LONLAT.transform(
        bike_df['x_act_ekf'].values, bike_df['y_act_ekf'].values
    )
    center_lat  = float(np.mean(lat_all))
    center_lon  = float(np.mean(lon_all))
    chain_str   = ' → '.join(chain_keys)
    movement_key = str(bike_df['movement_key'].dropna().iloc[0]) \
        if bike_df['movement_key'].notna().any() else '—'

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<title>Lane Debug — veh {veh_id}</title>
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Syne:wght@700&display=swap" rel="stylesheet"/>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
<style>
  :root {{
    --bg:#f8fafc; --panel:#ffffff; --border:#e2e8f0;
    --accent:#0891b2; --accent2:#ef4444; --accent3:#d97706;
    --text:#1e293b; --muted:#94a3b8;
    --font:'JetBrains Mono',monospace;
  }}
  *,*::before,*::after{{box-sizing:border-box;margin:0;padding:0;}}
  body{{background:var(--bg);color:var(--text);font-family:var(--font);
        font-size:12px;height:100vh;display:flex;flex-direction:column;overflow:hidden;}}

  #header{{display:flex;align-items:center;gap:12px;padding:7px 14px;
           border-bottom:1px solid var(--border);background:var(--panel);
           flex-shrink:0;flex-wrap:nowrap;}}
  #header h1{{font-family:'Syne',sans-serif;font-size:13px;color:var(--accent);white-space:nowrap;}}
  #chain-lbl{{font-size:10px;color:var(--muted);overflow:hidden;text-overflow:ellipsis;
              white-space:nowrap;flex:1;min-width:0;}}
  #seg-badge{{font-size:10px;padding:2px 8px;border-radius:20px;
              background:#f1f5f9;border:1px solid var(--accent);
              color:var(--accent);white-space:nowrap;flex-shrink:0;transition:all .2s;}}
  #info-bar{{display:flex;gap:14px;flex-shrink:0;font-size:10px;}}
  .ival{{color:var(--accent3);}}

  #main{{display:flex;flex:1;overflow:hidden;}}

  #map-panel{{width:50%;flex-shrink:0;border-right:1px solid var(--border);position:relative;}}
  #map{{width:100%;height:100%;}}

  #plot-panel{{flex:1;display:flex;flex-direction:column;overflow:hidden;}}
  .pb{{flex:1;border-bottom:1px solid var(--border);position:relative;min-height:0;}}
  .pb:last-child{{border-bottom:none;}}
  .plabel{{position:absolute;top:3px;left:8px;font-size:10px;color:var(--muted);
            letter-spacing:.08em;z-index:10;pointer-events:none;text-transform:uppercase;}}
  .pdiv{{width:100%;height:100%;}}

  #footer{{flex-shrink:0;padding:5px 14px;background:var(--panel);
           border-top:1px solid var(--border);display:flex;align-items:center;gap:10px;}}
  #play-btn{{background:none;border:1px solid var(--accent);color:var(--accent);
             font-family:var(--font);font-size:10px;padding:3px 10px;
             border-radius:4px;cursor:pointer;}}
  #play-btn:hover{{background:rgba(8,145,178,.08);}}
  #scrubber{{flex:1;accent-color:var(--accent);cursor:pointer;height:3px;}}
  #spd-sel{{background:var(--panel);border:1px solid var(--border);color:var(--text);
            font-family:var(--font);font-size:9px;padding:2px 5px;
            border-radius:4px;cursor:pointer;}}

  /* flag legend chips */
  #flag-legend{{display:flex;gap:8px;align-items:center;font-size:10px;color:var(--muted);flex-shrink:0;}}
  .flag-chip{{display:inline-flex;align-items:center;gap:4px;}}
  .flag-swatch{{width:12px;height:12px;border-radius:2px;opacity:.5;}}
  .flag-rev {{background:#ef4444;}}
  .flag-bike{{background:#22c55e;}}

  /* Leaflet legend */
  .legend{{background:rgba(255,255,255,.95);padding:7px 10px;border-radius:5px;
           border:1px solid #e2e8f0;font-size:10px;line-height:2.0;}}
  .li{{display:flex;align-items:center;gap:6px;}}
  .sw{{width:20px;height:3px;border-radius:1px;flex-shrink:0;}}
  .sw.band{{height:8px;opacity:.3;border-radius:2px;}}
  .sw.vpoly{{height:8px;opacity:.25;border-radius:2px;}}
</style>
</head>
<body>

<div id="header">
  <h1>⬡ Lane Debug — veh {veh_id}</h1>
  <div id="chain-lbl">{movement_key} &nbsp;|&nbsp; {chain_str}</div>
  <div id="seg-badge">–</div>
  <div id="info-bar">
    d=<span class="ival" id="dv">–</span>m
    s=<span class="ival" id="sv">–</span>m
    ṡ=<span class="ival" id="sdv">–</span>
    t=<span class="ival" id="tv">–</span>s
    #<span class="ival" id="fv">0</span>
  </div>
  <div id="flag-legend">
    <span class="flag-chip"><span class="flag-swatch flag-rev"></span>reverse</span>
    <span class="flag-chip"><span class="flag-swatch flag-bike"></span>in bike lane</span>
  </div>
</div>

<div id="main">
  <div id="map-panel"><div id="map"></div></div>
  <div id="plot-panel">
    <div class="pb"><div class="plabel">A · cumulative s vs d</div>            <div id="pa"  class="pdiv"></div></div>
    <div class="pb"><div class="plabel">B · s_native vs d_native (by seg)</div><div id="pb_" class="pdiv"></div></div>
    <div class="pb"><div class="plabel">C · speed · ṡ · ḋ vs time</div>       <div id="pc"  class="pdiv"></div></div>
  </div>
</div>

<div id="footer">
  <button id="play-btn">▶ PLAY</button>
  <input id="scrubber" type="range" min="0" max="100" value="0" step="1"/>
  <select id="spd-sel">
    <option value="1">1×</option><option value="2">2×</option>
    <option value="5">5×</option><option value="0.5">0.5×</option>
  </select>
</div>

<script>
// ── Embedded data ─────────────────────────────────────────────────────────
const TRAJ   = {traj_json};
const LAYERS = {layers_json};
const VRECTS = {vrects_json};
const N      = TRAJ.length;

// ── Colour palette (segment_id → colour, shared across all views) ─────────
const PAL = ['#0891b2','#f59e0b','#8b5cf6','#10b981','#ef4444','#f97316','#06b6d4','#84cc16'];
const segsAll = [...new Set(TRAJ.map(d=>d.seg_id).filter(Boolean))];
const SC={{}};
segsAll.forEach((s,i)=>SC[s]=PAL[i%PAL.length]);
const sc=s=>SC[s]||'#94a3b8';

// ── Leaflet ───────────────────────────────────────────────────────────────
const map=L.map('map',{{center:[{center_lat},{center_lon}],zoom:19}});
L.tileLayer(
  'https://wmts.geo.admin.ch/1.0.0/ch.swisstopo.swissimage/default/current/3857/{{z}}/{{x}}/{{y}}.jpeg',
  {{attribution:'© swisstopo',maxZoom:22,maxNativeZoom:21}}
).addTo(map);

// ── Layer groups (togglable) ──────────────────────────────────────────────
const lgCenterlines = L.layerGroup();
const lgValidity    = L.layerGroup();
const lgChangePoints= L.layerGroup();
const lgBikeBands   = L.layerGroup();
const lgTrajectory  = L.layerGroup();

const legItems=[];

LAYERS.forEach(lyr=>{{
  const col = sc(lyr.seg_key);

  // Validity polygon
  if(lyr.vp_ring && lyr.vp_ring.length > 2){{
    L.polygon(lyr.vp_ring, {{
      color: col, weight: 1, opacity: 0.5,
      fillColor: col, fillOpacity: 0.08,
      dashArray: '3 4'
    }}).bindTooltip(`validity: ${{lyr.seg_key}}`).addTo(lgValidity);
  }}

  // Change-point marker
  if(lyr.chg_pt){{
    L.circleMarker(lyr.chg_pt, {{
      radius: 5, color: '#1e293b', weight: 1.5,
      fillColor: col, fillOpacity: 0.9
    }}).bindTooltip(`s_change: ${{lyr.seg_key}}`).addTo(lgChangePoints);
    // Short tick line (just the marker is fine — avoids needing normal vector in JS)
  }}

  // Bike lane band
  if(lyr.bike_band){{
    L.polygon(lyr.bike_band, {{
      color:'#16a34a',weight:0,fillColor:'#22c55e',fillOpacity:.18
    }}).addTo(lgBikeBands);
    L.polyline(lyr.bike_far, {{color:'#16a34a',weight:1.5,opacity:.8,dashArray:'5 3'}})
     .bindTooltip(`outer edge: ${{lyr.seg_key}}`).addTo(lgBikeBands);
    L.polyline(lyr.bike_bnd, {{color:'#22c55e',weight:2,opacity:.9}})
     .bindTooltip(`inner boundary: ${{lyr.seg_key}} (side=${{lyr.side}}, w=${{lyr.w_bike}})`).addTo(lgBikeBands);
  }}

  // Centerline
  L.polyline(lyr.centerline,
    lyr.seg_type==='turn'
      ? {{color:col,weight:2,opacity:0.85,dashArray:'5 4'}}
      : {{color:col,weight:2.5,opacity:0.85}}
  ).bindTooltip(`centerline: ${{lyr.seg_key}}`).addTo(lgCenterlines);

  legItems.push(`<div class="li">
      <div class="sw" style="background:${{col}}"></div>
      ${{lyr.seg_key}}${{lyr.bike_band?'<span style="color:#22c55e"> ⬡</span>':''}}
    </div>`);
}});

// Trajectory polylines coloured by segment
let cseg=TRAJ[0].seg_id, pts=[];
const flush=(p,s)=>{{
  if(p.length<2) return;
  L.polyline(p,{{color:sc(s),weight:3.5,opacity:.70}}).addTo(lgTrajectory);
}};
for(const pt of TRAJ){{
  if(pt.seg_id!==cseg){{flush(pts,cseg);pts=[];cseg=pt.seg_id;}}
  pts.push([pt.lat,pt.lon]);
}}
flush(pts,cseg);

// Start / end markers
L.circleMarker([TRAJ[0].lat,TRAJ[0].lon],
  {{radius:6,color:'#fff',fillColor:'#0891b2',fillOpacity:1,weight:2}})
 .addTo(lgTrajectory).bindTooltip('START');
L.circleMarker([TRAJ[N-1].lat,TRAJ[N-1].lon],
  {{radius:6,color:'#fff',fillColor:'#ef4444',fillOpacity:1,weight:2}})
 .addTo(lgTrajectory).bindTooltip('END');

const liveM=L.circleMarker([TRAJ[0].lat,TRAJ[0].lon],
  {{radius:7,color:'#ffffff',fillColor:'#1e293b',fillOpacity:1,weight:3}}).addTo(map);

// Add all groups to map (trajectory always on, rest togglable)
lgTrajectory.addTo(map);
lgCenterlines.addTo(map);
lgValidity.addTo(map);
lgChangePoints.addTo(map);
lgBikeBands.addTo(map);

// Layer control
L.control.layers(null, {{
  'Centerlines':    lgCenterlines,
  'Validity polygons': lgValidity,
  'Change points':  lgChangePoints,
  'Bike lane bands':lgBikeBands,
  'Trajectory':     lgTrajectory,
}}, {{collapsed:false, position:'topright'}}).addTo(map);

// Legend (bottom-left)
const leg=L.control({{position:'bottomleft'}});
leg.onAdd=()=>{{
  const d=L.DomUtil.create('div','legend');
  d.innerHTML=`
    <div class="li"><div class="sw band" style="background:#22c55e"></div>Bike lane</div>
    <div class="li"><div class="sw vpoly" style="background:#94a3b8"></div>Validity polygon</div>
    ${{legItems.join('')}}`;
  return d;
}};
leg.addTo(map);

// ── Plotly base layout ────────────────────────────────────────────────────
const BG='#f8fafc', GR='#e2e8f0', TC='#94a3b8';
const bL={{
  paper_bgcolor:BG, plot_bgcolor:BG,
  font:{{family:'JetBrains Mono,monospace',size:10,color:TC}},
  margin:{{l:48,r:12,t:6,b:30}},
  xaxis:{{gridcolor:GR,zerolinecolor:'#cbd5e1',zerolinewidth:1,tickfont:{{size:10}}}},
  yaxis:{{gridcolor:GR,zerolinecolor:'#cbd5e1',zerolinewidth:1,tickfont:{{size:10}}}},
  showlegend:false, hovermode:'closest',
}};
const cfg={{responsive:true,displayModeBar:false}};

// ── Data arrays ───────────────────────────────────────────────────────────
const sarr    = TRAJ.map(d=>d.s);
const darr    = TRAJ.map(d=>d.d);
const snarr   = TRAJ.map(d=>d.s_native);
const dnarr   = TRAJ.map(d=>d.d_native);
const sdarr   = TRAJ.map(d=>d.s_dot);
const ddarr   = TRAJ.map(d=>d.d_dot);
const spd     = TRAJ.map(d=>d.speed);
const tarr    = TRAJ.map(d=>d.t);

// ── Continuous s across segment boundaries ───────────────────────────────
// s resets to 0 at each segment entry. Stitch by offsetting each segment so
// it starts where the previous one ended → monotone cumulative-s axis.
const csarr = (()=>{{
  const out = new Array(N).fill(null);
  let offset = 0;
  let prevSeg = null;
  for(let i=0;i<N;i++){{
    const si  = sarr[i];
    const seg = TRAJ[i].seg_id;
    if(si===null||si===undefined||si!==si) continue;  // null / NaN guard
    if(seg !== prevSeg && prevSeg !== null){{
      // Segment boundary: advance offset by the last valid cumulative value
      // so the new segment's s=0 maps to that position.
      let last=null;
      for(let j=i-1;j>=0;j--){{if(out[j]!==null){{last=out[j];break;}}}}
      if(last!==null) offset=last;
    }}
    out[i] = offset + si;
    prevSeg = seg;
  }}
  return out;
}})();

// ── Build per-segment traces helper ──────────────────────────────────────
function mkTraces(xA, yA, addCursor=true){{
  const tr=[];
  segsAll.forEach(seg=>{{
    const idx=TRAJ.map((_,i)=>TRAJ[i].seg_id===seg?i:-1).filter(i=>i>=0);
    if(!idx.length) return;
    tr.push({{
      x:idx.map(i=>xA[i]), y:idx.map(i=>yA[i]),
      mode:'lines', line:{{color:sc(seg),width:1.8}}, name:seg,
    }});
  }});
  // Unmatched
  const ui=TRAJ.map((_,i)=>!TRAJ[i].seg_id?i:-1).filter(i=>i>=0);
  if(ui.length) tr.push({{
    x:ui.map(i=>xA[i]), y:ui.map(i=>yA[i]),
    mode:'lines', line:{{color:'#94a3b8',width:1,dash:'dot'}}, name:'unmatched',
  }});
  // Cursor marker
  if(addCursor) tr.push({{
    x:[xA[0]], y:[yA[0]], mode:'markers',
    marker:{{color:'#1e293b',size:10,symbol:'circle',line:{{color:'#ffffff',width:2}}}},
    name:'cursor',
  }});
  return tr;
}}

function mkLayout(xtitle, ytitle, extraShapes=[]){{
  return Object.assign({{}}, bL, {{
    xaxis: Object.assign({{}}, bL.xaxis, {{title:{{text:xtitle,font:{{size:10}}}}}}),
    yaxis: Object.assign({{}}, bL.yaxis, {{title:{{text:ytitle,font:{{size:10}}}},zeroline:true}}),
    shapes: extraShapes,
  }});
}}

// ── Plot A: cumulative s vs d  (vrect flags on s-axis via time mapping) ───
// For vrects on A we map time ranges → cumulative-s ranges
function timeToCS(t){{
  const i=tarr.reduce((b,v,i)=>Math.abs(v-t)<Math.abs(tarr[b]-t)?i:b,0);
  return csarr[i];
}}
const shapesA_rev = VRECTS.reverse.map(sh=>Object.assign({{}},sh,{{
  xref:'x',
  x0: timeToCS(sh.x0)||0,
  x1: timeToCS(sh.x1)||0,
}}));
const shapesA_bk  = VRECTS.bike_lane.map(sh=>Object.assign({{}},sh,{{
  xref:'x',
  x0: timeToCS(sh.x0)||0,
  x1: timeToCS(sh.x1)||0,
}}));

const tA = mkTraces(csarr, darr);
Plotly.newPlot('pa', tA,
  mkLayout('cumulative s [m]','d [m]', [...shapesA_rev, ...shapesA_bk]),
  cfg);

// ── Plot B: s_native vs d_native, coloured by segment ────────────────────
// No vrects here — time is animation dim only (cursor moves, no x=time axis)
const tB = mkTraces(snarr, dnarr);
Plotly.newPlot('pb_', tB,
  mkLayout('s_native [m]','d_native [m]'),
  cfg);

// ── Plot C: speed_ekf + s_dot + d_dot vs time ────────────────────────────
// Colours per segment; legend shows only 3 fixed signal-type entries.
// Segment colour is readable from the map legend — no need to repeat here.
const shapesC = [...VRECTS.reverse, ...VRECTS.bike_lane];

const tC = [];
// speed_ekf — single grey dashed trace
tC.push({{x:tarr, y:spd,
  mode:'lines', line:{{color:'#94a3b8',width:2,dash:'dash'}},
  name:'|v| speed', legendgroup:'spd',
  hovertemplate:'t=%{{x:.2f}}s  |v|=%{{y:.2f}} m/s<extra></extra>'}});
// s_dot and d_dot per segment — only first segment contributes legend entry
segsAll.forEach((seg,si)=>{{
  const idx=TRAJ.map((_,i)=>TRAJ[i].seg_id===seg?i:-1).filter(i=>i>=0);
  if(!idx.length) return;
  const col=sc(seg);
  tC.push({{x:idx.map(i=>tarr[i]), y:idx.map(i=>sdarr[i]),
    mode:'lines', line:{{color:col,width:1.8}},
    legendgroup:'sdot', showlegend:si===0, name:'ṡ s_dot',
    hovertemplate:`${{seg}}<br>t=%{{x:.2f}}s  ṡ=%{{y:.2f}} m/s<extra></extra>`}});
  tC.push({{x:idx.map(i=>tarr[i]), y:idx.map(i=>ddarr[i]),
    mode:'lines', line:{{color:col,width:1.2,dash:'dot'}},
    legendgroup:'ddot', showlegend:si===0, name:'ḋ d_dot',
    hovertemplate:`${{seg}}<br>t=%{{x:.2f}}s  ḋ=%{{y:.2f}} m/s<extra></extra>`}});
}});
// Unmatched
const uiC=TRAJ.map((_,i)=>!TRAJ[i].seg_id?i:-1).filter(i=>i>=0);
if(uiC.length){{
  tC.push({{x:uiC.map(i=>tarr[i]),y:uiC.map(i=>sdarr[i]),mode:'lines',
    line:{{color:'#94a3b8',width:1}},legendgroup:'sdot',showlegend:false,name:'ṡ unmatched'}});
  tC.push({{x:uiC.map(i=>tarr[i]),y:uiC.map(i=>ddarr[i]),mode:'lines',
    line:{{color:'#94a3b8',width:0.8}},legendgroup:'ddot',showlegend:false,name:'ḋ unmatched'}});
}}
// cursor
tC.push({{x:[tarr[0]], y:[spd[0]], mode:'markers',
  marker:{{color:'#1e293b',size:10,symbol:'circle',line:{{color:'#ffffff',width:2}}}},
  name:'cursor', showlegend:true}});

const layoutC = Object.assign({{}}, bL, {{
  xaxis: Object.assign({{}}, bL.xaxis, {{title:{{text:'t [s]',font:{{size:10}}}}}}),
  yaxis: Object.assign({{}}, bL.yaxis, {{title:{{text:'m/s',font:{{size:10}}}},zeroline:true}}),
  shapes: shapesC,
  showlegend: true,
  legend: {{orientation:'h',x:0,y:-0.15,xanchor:'left',yanchor:'top',
            font:{{size:10}},bgcolor:'rgba(248,250,252,0.88)',
            bordercolor:'#e2e8f0',borderwidth:1}},
  margin: {{l:48,r:12,t:6,b:70}},
}});
Plotly.newPlot('pc', tC, layoutC, cfg);

const cA = tA.length-1;   // cursor trace index in each plot
const cB = tB.length-1;
const cC = tC.length-1;

// ── Shared update ─────────────────────────────────────────────────────────
let cur=0;
function upd(i){{
  i=Math.max(0,Math.min(N-1,Math.round(i)));
  cur=i;
  const pt=TRAJ[i];

  liveM.setLatLng([pt.lat,pt.lon]);

  document.getElementById('dv').textContent  = pt.d     !==null ? pt.d.toFixed(2)     : '–';
  document.getElementById('sv').textContent  = pt.s     !==null ? pt.s.toFixed(1)     : '–';
  document.getElementById('sdv').textContent = pt.s_dot !==null ? pt.s_dot.toFixed(2) : '–';
  document.getElementById('tv').textContent  = pt.t     !==null ? pt.t.toFixed(2)     : '–';
  document.getElementById('fv').textContent  = i;
  document.getElementById('scrubber').value  = i;

  const badge = document.getElementById('seg-badge');
  badge.textContent   = pt.seg_id ? `${{pt.seg_role?.toUpperCase()}} · ${{pt.seg_id}}` : 'UNMATCHED';
  badge.style.borderColor = pt.seg_id ? sc(pt.seg_id) : '#94a3b8';
  badge.style.color       = pt.seg_id ? sc(pt.seg_id) : '#94a3b8';

  // Move cursors
  Plotly.restyle('pa',  {{x:[[csarr[i]]],     y:[[darr[i]]]}},  [cA]);
  Plotly.restyle('pb_', {{x:[[snarr[i]]],     y:[[dnarr[i]]]}}, [cB]);
  Plotly.restyle('pc',  {{x:[[tarr[i]]],      y:[[spd[i]]]}},   [cC]);
}}

// ── Click-to-scrub ────────────────────────────────────────────────────────
document.getElementById('pa').on('plotly_click', data=>{{
  const x=data.points[0].x;
  upd(csarr.reduce((b,v,i)=>v!==null&&Math.abs(v-x)<Math.abs((csarr[b]||0)-x)?i:b,0));
}});
document.getElementById('pb_').on('plotly_click', data=>{{
  const x=data.points[0].x;
  upd(snarr.reduce((b,v,i)=>v!==null&&Math.abs(v-x)<Math.abs((snarr[b]||0)-x)?i:b,0));
}});
document.getElementById('pc').on('plotly_click', data=>{{
  const x=data.points[0].x;
  upd(tarr.reduce((b,v,i)=>Math.abs(v-x)<Math.abs(tarr[b]-x)?i:b,0));
}});

document.getElementById('scrubber').max=N-1;
document.getElementById('scrubber').addEventListener('input',function(){{upd(parseInt(this.value));}});

// ── Playback ──────────────────────────────────────────────────────────────
let playing=false,timer=null;
function startPlay(){{
  playing=true;
  document.getElementById('play-btn').textContent='⏸ PAUSE';
  const mul=parseFloat(document.getElementById('spd-sel').value);
  timer=setInterval(()=>{{if(cur>=N-1){{stopPlay();return;}}upd(cur+1);}},1000/(20*mul));
}}
function stopPlay(){{
  playing=false; clearInterval(timer);
  document.getElementById('play-btn').textContent='▶ PLAY';
}}
document.getElementById('play-btn').addEventListener('click',()=>playing?stopPlay():startPlay());
document.getElementById('spd-sel').addEventListener('change',()=>{{if(playing){{stopPlay();startPlay();}}}});

upd(0);
</script>
</body>
</html>"""

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f'Saved: {output_path}  ({len(bike_df)} frames, {len(chain_keys)} segments: {chain_str})')
    return output_path