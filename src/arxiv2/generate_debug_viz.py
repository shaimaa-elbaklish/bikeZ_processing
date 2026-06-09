"""
debug_bikelane_map.py
---------------------
Interactive HTML debug visualisation for one bicycle trajectory after
lane coordinate transform.

LEFT  pane : Leaflet / swisstopo satellite map
  - All centerlines in the matched chain (approach, turn, departure)
  - Bike lane band for every lane segment that has one
  - Trajectory coloured by segment
  - Animated playhead

RIGHT pane : Five Plotly time-series stacked vertically
  A  cumulative distance vs d
  B  native s vs d
  C  s_dot vs time
  D  d_dot vs time
  E  speed vs time

Usage
-----
    from debug_bikelane_map import generate_bikelane_debug_map
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
from scipy.interpolate import splev
from pyproj import Transformer


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_to_wgs84 = Transformer.from_crs("EPSG:2056", "EPSG:4326", always_xy=True)


def _sd_to_latlon(s_arr, d_arr, tck, unew, cum_dist, x_offset, y_offset):
    """(s_native, d) pairs on a spline → [[lat, lon], ...] in WGS84."""
    s_arr = np.asarray(s_arr, dtype=float)
    d_arr = np.asarray(d_arr, dtype=float)
    t_arr    = np.interp(s_arr, cum_dist, unew)
    xp, yp   = splev(t_arr, tck, der=0)
    xp1, yp1 = splev(t_arr, tck, der=1)
    norms    = np.sqrt(xp1**2 + yp1**2)
    norms    = np.where(norms > 1e-12, norms, 1.0)
    tx = xp1 / norms;  ty = yp1 / norms
    nx = -ty;          ny =  tx          # left-hand normal
    x_pts = xp + d_arr * nx
    y_pts = yp + d_arr * ny
    lon, lat = _to_wgs84.transform(x_pts + x_offset, y_pts + y_offset)
    return [[float(la), float(lo)] for la, lo in zip(lat, lon)]


def _safe(val):
    try:
        if math.isnan(val) or math.isinf(val):
            return None
    except (TypeError, ValueError):
        pass
    return val


def _build_traj_json(df):
    """Serialise trajectory to JSON; lat/lon from x_act_ekf/y_act_ekf."""
    lon_arr, lat_arr = _to_wgs84.transform(
        df["x_act_ekf"].values, df["y_act_ekf"].values
    )
    cum_dist = np.concatenate([[0], np.cumsum(
        np.linalg.norm(np.diff(df[["x_ekf", "y_ekf"]].to_numpy(), axis=0), axis=1)
    )])
    records = []
    for i, row in enumerate(df.itertuples(index=False)):
        records.append({
            "t":        float(row.time),
            "lat":      float(lat_arr[i]),
            "lon":      float(lon_arr[i]),
            "cum_dist": float(cum_dist[i]),
            "s":        _safe(row.s),
            "d":        _safe(row.d),
            "s_dot":    _safe(row.s_dot),
            "d_dot":    _safe(row.d_dot),
            "speed":    _safe(row.speed_ekf),
            "seg_id":   str(row.segment_id)   if row.segment_id   is not None else None,
            "seg_role": str(row.segment_role) if row.segment_role is not None else None,
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
    For every segment in the chain build centerline + optional bike lane band.
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
        tck, unew, cum_dist = geometry_store[geom_key]['spline']
        total_len = geometry_store[geom_key]['total_length']

        # Full centerline
        s_cl  = np.linspace(0, total_len, n_pts)
        cl_ll = _sd_to_latlon(s_cl, np.zeros(n_pts),
                               tck, unew, cum_dist, x_offset, y_offset)

        layer = {
            "seg_key":    seg_key,
            "seg_type":   entry["type"],
            "centerline": cl_ll,
            "bike_band":  None,
            "bike_bnd":   None,
            "bike_far":   None,
            "w_bike":     None,
            "side":       None,
        }

        # Bike lane band — lane segments only
        bike_lane = entry.get("bike_lane")
        if (entry["type"] == "lane" and bike_lane is not None
                and "d_boundary_spline" in bike_lane):
            d_bnd_spl        = bike_lane["d_boundary_spline"]
            w_bike           = bike_lane["w_bike"]
            side             = bike_lane["side"]
            s_min, s_max     = bike_lane["s_domain"]

            s_bl  = np.linspace(s_min, s_max, n_pts)
            d_bnd = d_bnd_spl(s_bl)
            d_far = d_bnd + side * w_bike

            bnd_ll = _sd_to_latlon(s_bl, d_bnd, tck, unew, cum_dist,
                                    x_offset, y_offset)
            far_ll = _sd_to_latlon(s_bl, d_far, tck, unew, cum_dist,
                                    x_offset, y_offset)

            layer["bike_bnd"]  = bnd_ll
            layer["bike_far"]  = far_ll
            layer["bike_band"] = bnd_ll + far_ll[::-1] + [bnd_ll[0]]
            layer["w_bike"]    = float(w_bike)
            layer["side"]      = int(side)

        layers.append(layer)

    return layers


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

    lon_all, lat_all = _to_wgs84.transform(
        bike_df['x_act_ekf'].values, bike_df['y_act_ekf'].values
    )
    center_lat  = float(np.mean(lat_all))
    center_lon  = float(np.mean(lon_all))
    chain_str   = ' → '.join(chain_keys)
    movement_key = str(bike_df['movement_key'].dropna().iloc[0]) \
        if bike_df['movement_key'].notna().any() else '—'

    
    # For dark mode:
    # :root {{
    #   --bg:#0d0f14; --panel:#13161e; --border:#1f2433;
    #   --accent:#00e5c0; --accent2:#ff6b6b; --accent3:#ffd166;
    #   --text:#c8cdd8; --muted:#4a5068;
    #   --font:'JetBrains Mono',monospace;
    # }}
    # // ── Plotly helpers ────────────────────────────────────────────────────────
    # const BG='#0d0f14',GR='#1f2433',TC='#4a5068';
    # .legend{{background:rgba(13,15,20,.9);padding:7px 10px;border-radius:5px;
    #          border:1px solid #1f2433;font-size:8px;line-height:2;}}
    
    # For light mode:
    # :root {{
    #   --bg:#f8fafc; --panel:#ffffff; --border:#e2e8f0;
    #   --accent:#0891b2; --accent2:#ef4444; --accent3:#d97706;
    #   --text:#1e293b; --muted:#94a3b8;
    #   --font:'JetBrains Mono',monospace;
    # }}
    # // ── Plotly helpers ────────────────────────────────────────────────────────
    # const BG='#f8fafc', GR='#e2e8f0', TC='#94a3b8';
    # .legend{{background:rgba(248,250,252,.95);padding:7px 10px;border-radius:5px;
    #      border:1px solid #e2e8f0;font-size:8px;line-height:2;}}

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
           border-bottom:1px solid var(--border);background:var(--panel);flex-shrink:0;flex-wrap:nowrap;}}
  #header h1{{font-family:'Syne',sans-serif;font-size:13px;color:var(--accent);white-space:nowrap;}}
  #chain-lbl{{font-size:8px;color:var(--muted);overflow:hidden;text-overflow:ellipsis;
              white-space:nowrap;flex:1;min-width:0;}}
  #seg-badge{{font-size:8px;padding:2px 8px;border-radius:20px;background:#1a2035;
              border:1px solid var(--accent);color:var(--accent);white-space:nowrap;flex-shrink:0;
              transition:all .2s;}}
  #info-bar{{display:flex;gap:14px;flex-shrink:0;font-size:9px;}}
  .ival{{color:var(--accent3);}}

  #main{{display:flex;flex:1;overflow:hidden;}}

  #map-panel{{width:50%;flex-shrink:0;border-right:1px solid var(--border);position:relative;}}
  #map{{width:100%;height:100%;}}

  #plot-panel{{flex:1;display:flex;flex-direction:column;overflow:hidden;}}
  .pb{{flex:1;border-bottom:1px solid var(--border);position:relative;min-height:0;}}
  .pb:last-child{{border-bottom:none;}}
  .plabel{{position:absolute;top:3px;left:8px;font-size:7px;color:var(--muted);
            letter-spacing:.08em;z-index:10;pointer-events:none;text-transform:uppercase;}}
  .pdiv{{width:100%;height:100%;}}

  #footer{{flex-shrink:0;padding:5px 14px;background:var(--panel);
           border-top:1px solid var(--border);display:flex;align-items:center;gap:10px;}}
  #play-btn{{background:none;border:1px solid var(--accent);color:var(--accent);
             font-family:var(--font);font-size:10px;padding:3px 10px;
             border-radius:4px;cursor:pointer;}}
  #play-btn:hover{{background:rgba(0,229,192,.1);}}
  #scrubber{{flex:1;accent-color:var(--accent);cursor:pointer;height:3px;}}
  #spd-sel{{background:var(--panel);border:1px solid var(--border);color:var(--text);
            font-family:var(--font);font-size:9px;padding:2px 5px;border-radius:4px;cursor:pointer;}}

  .legend{{background:rgba(248,250,252,.95);padding:7px 10px;border-radius:5px;
         border:1px solid #e2e8f0;font-size:8px;line-height:2;}}
  .li{{display:flex;align-items:center;gap:6px;}}
  .sw{{width:20px;height:3px;border-radius:1px;flex-shrink:0;}}
  .sw.band{{height:8px;opacity:.3;border-radius:2px;}}
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
</div>

<div id="main">
  <div id="map-panel"><div id="map"></div></div>
  <div id="plot-panel">
    <div class="pb"><div class="plabel">A · cumulative dist vs d</div>   <div id="pa" class="pdiv"></div></div>
    <div class="pb"><div class="plabel">B · s vs d</div>                 <div id="pb_" class="pdiv"></div></div>
    <div class="pb"><div class="plabel">C · ṡ longitudinal speed vs t</div><div id="pc" class="pdiv"></div></div>
    <div class="pb"><div class="plabel">D · ḋ lateral speed vs t</div>  <div id="pd_" class="pdiv"></div></div>
    <div class="pb"><div class="plabel">E · speed |v| vs t</div>         <div id="pe" class="pdiv"></div></div>
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
const N      = TRAJ.length;

// ── Colour palette ────────────────────────────────────────────────────────
const PAL = ['#00e5c0','#ff6b6b','#ffd166','#a78bfa','#60a5fa','#fb923c','#34d399'];
const segsAll = [...new Set(TRAJ.map(d=>d.seg_id).filter(Boolean))];
const SC={{}};
segsAll.forEach((s,i)=>SC[s]=PAL[i%PAL.length]);
const sc=s=>SC[s]||'#4a5068';

// ── Leaflet ───────────────────────────────────────────────────────────────
const map=L.map('map',{{center:[{center_lat},{center_lon}],zoom:19}});
L.tileLayer(
  'https://wmts.geo.admin.ch/1.0.0/ch.swisstopo.swissimage/default/current/3857/{{z}}/{{x}}/{{y}}.jpeg',
  {{attribution:'© swisstopo',maxZoom:22,maxNativeZoom:21}}
).addTo(map);

// Draw chain layers (back to front: band → far → bnd → centerline)
const legItems=[];
LAYERS.forEach(lyr=>{{
  const col=sc(lyr.seg_key);
  // bike lane band
  if(lyr.bike_band){{
    L.polygon(lyr.bike_band,{{color:'#22c55e',weight:0,fillColor:'#22c55e',fillOpacity:.18}}).addTo(map);
    L.polyline(lyr.bike_far,{{color:'#16a34a',weight:1.5,opacity:.8,dashArray:'5 3'}})
     .addTo(map).bindTooltip(`outer edge: ${{lyr.seg_key}}`);
    L.polyline(lyr.bike_bnd,{{color:'#22c55e',weight:2,opacity:.9}})
     .addTo(map).bindTooltip(`inner boundary: ${{lyr.seg_key}} (side=${{lyr.side}}, w=${{lyr.w_bike}}m)`);
  }}
  // centerline
  L.polyline(lyr.centerline,
    lyr.seg_type==='turn'
      ? {{color:col, weight:2,   opacity:0.8, dashArray:'4 4'}}
      : {{color:col, weight:2,   opacity:0.8, dashArray:'4 4'}}
  ).addTo(map).bindTooltip(`centerline: ${{lyr.seg_key}}`);

  legItems.push(`<div class="li">
      <div class="sw" style="background:${{col}}"></div>
      ${{lyr.seg_key}}${{lyr.bike_band ? `<span style="color:#22c55e"> ⬡</span>` : ''}}
    </div>
    <div class="li">
      <div class="sw" style="background:repeating-linear-gradient(90deg,${{col}} 0px,${{col}} 4px,transparent 4px,transparent 8px)"></div>
      <span style="color:var(--muted);font-size:7px">centerline</span>
    </div>`);
}});

// Trajectory polylines coloured by segment
let cseg=TRAJ[0].seg_id,pts=[];
const flush=(p,s)=>{{if(p.length<2)return;
  L.polyline(p,{{color:sc(s),weight:3.5,opacity:.65}}).addTo(map);}};
for(const pt of TRAJ){{
  if(pt.seg_id!==cseg){{flush(pts,cseg);pts=[];cseg=pt.seg_id;}}
  pts.push([pt.lat,pt.lon]);
}}
flush(pts,cseg);

L.circleMarker([TRAJ[0].lat,TRAJ[0].lon],
  {{radius:6,color:'#fff',fillColor:'#00e5c0',fillOpacity:1,weight:2}})
 .addTo(map).bindTooltip('START');
L.circleMarker([TRAJ[N-1].lat,TRAJ[N-1].lon],
  {{radius:6,color:'#fff',fillColor:'#ff6b6b',fillOpacity:1,weight:2}})
 .addTo(map).bindTooltip('END');

const liveM=L.circleMarker([TRAJ[0].lat,TRAJ[0].lon],
  {{radius:7,color:'#00e5c0',fillColor:'#fff',fillOpacity:1,weight:3}}).addTo(map);

// Legend
const leg=L.control({{position:'bottomleft'}});
leg.onAdd=()=>{{
  const d=L.DomUtil.create('div','legend');
  d.innerHTML=`
      <div class="li"><div class="sw band" style="background:#22c55e"></div>Bike lane band</div>
      ${{legItems.join('')}}`;
  return d;
}};
leg.addTo(map);

// ── Plotly helpers ────────────────────────────────────────────────────────
const BG='#f8fafc', GR='#e2e8f0', TC='#94a3b8';
const bL={{
  paper_bgcolor:BG,plot_bgcolor:BG,
  font:{{family:'JetBrains Mono,monospace',size:8,color:TC}},
  margin:{{l:40,r:8,t:4,b:26}},
  xaxis:{{gridcolor:GR,zerolinecolor:'#2a2f40',zerolinewidth:1}},
  yaxis:{{gridcolor:GR,zerolinecolor:'#2a2f40',zerolinewidth:1}},
  showlegend:false,hovermode:'closest',
}};
const cfg={{responsive:true,displayModeBar:false}};

// Arrays
const cumd =TRAJ.map(d=>d.cum_dist);
const darr =TRAJ.map(d=>d.d);
const sarr =TRAJ.map(d=>d.s);
const sdarr=TRAJ.map(d=>d.s_dot);
const ddarr=TRAJ.map(d=>d.d_dot);
const spd  =TRAJ.map(d=>d.speed);
const tarr =TRAJ.map(d=>d.t);

// Build one trace per segment + cursor
function mkTraces(xA,yA){{
  const tr=[];
  segsAll.forEach(seg=>{{
    const idx=TRAJ.map((_,i)=>TRAJ[i].seg_id===seg?i:-1).filter(i=>i>=0);
    tr.push({{x:idx.map(i=>xA[i]),y:idx.map(i=>yA[i]),
              mode:'lines',line:{{color:sc(seg),width:1.8}},name:seg}});
  }});
  const ui=TRAJ.map((_,i)=>!TRAJ[i].seg_id?i:-1).filter(i=>i>=0);
  if(ui.length) tr.push({{x:ui.map(i=>xA[i]),y:ui.map(i=>yA[i]),
    mode:'lines',line:{{color:'#2a2f40',width:1,dash:'dot'}},name:'unmatched'}});
  tr.push({{x:[xA[0]],y:[yA[0]],mode:'markers',
    marker:{{color:'#fff',size:7,symbol:'circle',line:{{color:'#00e5c0',width:2}}}},
    name:'cursor'}});
  return tr;
}}

function mkLayout(xtitle,ytitle){{
  return Object.assign({{}},bL,{{
    xaxis:Object.assign({{}},bL.xaxis,{{title:{{text:xtitle,font:{{size:8}}}}}}) ,
    yaxis:Object.assign({{}},bL.yaxis,{{title:{{text:ytitle,font:{{size:8}}}},zeroline:true}}),
  }});
}}

const tA=mkTraces(cumd,darr);  Plotly.newPlot('pa',  tA, mkLayout('cum dist [m]','d [m]'),   cfg);
const tB=mkTraces(sarr,darr);  Plotly.newPlot('pb_', tB, mkLayout('s [m]',       'd [m]'),   cfg);
const tC=mkTraces(tarr,sdarr); Plotly.newPlot('pc',  tC, mkLayout('t [s]',       'ṡ [m/s]'), cfg);
const tD=mkTraces(tarr,ddarr); Plotly.newPlot('pd_', tD, mkLayout('t [s]',       'ḋ [m/s]'), cfg);
const tE=mkTraces(tarr,spd);   Plotly.newPlot('pe',  tE, mkLayout('t [s]',       '|v| [m/s]'),cfg);

const cA=tA.length-1, cB=tB.length-1, cC=tC.length-1, cD=tD.length-1, cE=tE.length-1;

// ── Shared update ─────────────────────────────────────────────────────────
let cur=0;
function upd(i){{
  i=Math.max(0,Math.min(N-1,Math.round(i)));
  cur=i;
  const pt=TRAJ[i];
  liveM.setLatLng([pt.lat,pt.lon]);
  document.getElementById('dv').textContent  =pt.d    !==null?pt.d.toFixed(2):'–';
  document.getElementById('sv').textContent  =pt.s    !==null?pt.s.toFixed(1):'–';
  document.getElementById('sdv').textContent =pt.s_dot!==null?pt.s_dot.toFixed(2):'–';
  document.getElementById('tv').textContent  =pt.t    !==null?pt.t.toFixed(2):'–';
  document.getElementById('fv').textContent  =i;
  document.getElementById('scrubber').value  =i;
  const badge=document.getElementById('seg-badge');
  badge.textContent  =pt.seg_id?`${{pt.seg_role?.toUpperCase()}} · ${{pt.seg_id}}`:'UNMATCHED';
  badge.style.borderColor=pt.seg_id?sc(pt.seg_id):'#2a2f40';
  badge.style.color      =pt.seg_id?sc(pt.seg_id):'#4a5068';
  const uc=(div,ci,xA,yA)=>Plotly.restyle(div,{{x:[[xA[i]]],y:[[yA[i]]]}},  [ci]);
  uc('pa', cA,cumd,darr);
  uc('pb_',cB,sarr,darr);
  uc('pc', cC,tarr,sdarr);
  uc('pd_',cD,tarr,ddarr);
  uc('pe', cE,tarr,spd);
}}

// Click-to-scrub: cum dist axis
document.getElementById('pa').on('plotly_click',data=>{{
  const x=data.points[0].x;
  upd(cumd.reduce((b,v,i)=>Math.abs(v-x)<Math.abs(cumd[b]-x)?i:b,0));
}});
// Click-to-scrub: time-axis plots
['pc','pd_','pe'].forEach(id=>{{
  document.getElementById(id).on('plotly_click',data=>{{
    const x=data.points[0].x;
    upd(tarr.reduce((b,v,i)=>Math.abs(v-x)<Math.abs(tarr[b]-x)?i:b,0));
  }});
}});

document.getElementById('scrubber').max=N-1;
document.getElementById('scrubber').addEventListener('input',function(){{upd(parseInt(this.value));}});

let playing=false,timer=null;
function startPlay(){{
  playing=true;
  document.getElementById('play-btn').textContent='⏸ PAUSE';
  const mul=parseFloat(document.getElementById('spd-sel').value);
  timer=setInterval(()=>{{if(cur>=N-1){{stopPlay();return;}}upd(cur+1);}},1000/(20*mul));
}}
function stopPlay(){{
  playing=false;clearInterval(timer);
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