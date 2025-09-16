#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI-driven crop lifecycle monitoring & yield forecasting (county-level, USA)

Features
--------
1) Downloads vegetation indices by county & year:
   - MODIS/061/MOD13Q1: NDVI + EVI (16-day, 250m)
   - Landsat 8/9 L2: NDVI (16-day bins, 30m)
   - BOTH: build both and also a COMBINED provider (averaged tabular, aligned sequence)

2) Masks to crop using USDA/NASS/CDL (best-effort fallback if missing).

3) Builds per-year sequences (DOY ~80..320) & summary features:
   - NDVI/EVI AUC, peak, mean, timing-of-peak.

4) Fetches USDA NASS county yields (robust retries, name variants).

5) Models:
   - BaselineMean, Linear Regression, Random Forest (tabular features with robust imputation)
   - BiLSTM + Attention on sequences (with DOY sin/cos features)

6) Evaluation:
   - Leave-One-Year-Out (LOYO) CV per provider (MODIS/LANDSAT/COMBINED)
   - Final hold-out test on the last labeled year

7) Saves:
   - crop_yield_results/ts_all.csv
   - crop_yield_results/features_all.csv
   - crop_yield_results/cv_metrics.csv
   - crop_yield_results/final_test_predictions.csv
   - crop_yield_results/model_comparison.csv

Usage (examples)
----------------
export EE_PROJECT="your-ee-project"
export NASS_API_KEY="xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"

# Specific counties, both providers
python3 crop_yield.py \
  --state IA \
  --counties "Story;Polk;Boone" \
  --crop CORN \
  --start_year 2016 --end_year 2023 \
  --provider BOTH \
  --epochs 50 \
  --max_workers 4

# All counties in a state (optionally exclude some)
python3 crop_yield.py \
  --state IA \
  --all_counties \
  --exclude_counties "Polk" \
  --crop CORN \
  --start_year 2003 --end_year 2023 \
  --provider MODIS \
  --epochs 150 \
  --max_workers 8
"""
import os, sys, json, argparse, warnings, shutil, subprocess, uuid, time, math, threading
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import requests

# ML
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer

# Plot (optional; not used by default)
import matplotlib.pyplot as plt

# Parallel
from concurrent.futures import ThreadPoolExecutor, as_completed

# Earth Engine
try:
    import ee
    import geemap  # noqa: F401
except Exception:
    print("ERROR: earthengine-api/geemap not installed.\n"
          "  pip install earthengine-api geemap pandas numpy requests scikit-learn tensorflow matplotlib")
    sys.exit(1)

OUT_DIR = "crop_yield_results"
TS_CSV  = os.path.join(OUT_DIR, "ts_all.csv")
FEAT_CSV= os.path.join(OUT_DIR, "features_all.csv")
CV_CSV  = os.path.join(OUT_DIR, "cv_metrics.csv")
TEST_PRED_CSV = os.path.join(OUT_DIR, "final_test_predictions.csv")
COMPARE_CSV   = os.path.join(OUT_DIR, "model_comparison.csv")

# -------------------------- Utils --------------------------
def ensure_dir(d): os.makedirs(d, exist_ok=True)
def _run(cmd, check=True):
    r = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if check and r.returncode != 0:
        raise RuntimeError(f"Command failed ({r.returncode}): {cmd}\nOutput:\n{r.stdout}")
    return r.stdout.strip()
def _which(x): return shutil.which(x) is not None

def trapezoid_auc(y_vals, x_step=16):
    y = np.asarray(y_vals, float)
    if len(y) < 2: return float(np.nan)
    return float(np.trapz(y, dx=x_step))

def timing_of_peak(y_vals, doy_vals):
    if len(y_vals) == 0: return np.nan
    y = np.asarray(y_vals, float)
    if not np.isfinite(y).any(): return np.nan
    i = int(np.nanargmax(y))
    return float(doy_vals[i]) if i < len(doy_vals) else np.nan

def state_to_fips(state_abbr):
    table = {'AL':'01','AK':'02','AZ':'04','AR':'05','CA':'06','CO':'08','CT':'09','DE':'10','DC':'11','FL':'12',
        'GA':'13','HI':'15','ID':'16','IL':'17','IN':'18','IA':'19','KS':'20','KY':'21','LA':'22','ME':'23',
        'MD':'24','MA':'25','MI':'26','MN':'27','MS':'28','MO':'29','MT':'30','NE':'31','NV':'32','NH':'33',
        'NJ':'34','NM':'35','NY':'36','NC':'37','ND':'38','OH':'39','OK':'40','OR':'41','PA':'42','RI':'44',
        'SC':'45','SD':'46','TN':'47','TX':'48','UT':'49','VT':'50','VA':'51','WA':'53','WV':'54','WI':'55','WY':'56'}
    return table[state_abbr.upper()]

def get_county_geom(state_abbr, county_name):
    statefp = state_to_fips(state_abbr)
    fc = ee.FeatureCollection("TIGER/2018/Counties") \
            .filter(ee.Filter.eq("STATEFP", statefp)) \
            .filter(ee.Filter.eq("NAME", county_name))
    f = fc.first()
    if f is None:
        raise RuntimeError(f"County not found: {county_name}, {state_abbr}")
    geoid = f.get("GEOID").getInfo()
    geom = f.geometry()
    return geom, geoid

def list_state_counties(state_abbr):
    statefp = state_to_fips(state_abbr)
    fc = ee.FeatureCollection("TIGER/2018/Counties").filter(ee.Filter.eq("STATEFP", statefp))
    names = fc.aggregate_array("NAME").getInfo()
    names = sorted(list(dict.fromkeys([str(n) for n in names])))
    return names

# ---------------------- Earth Engine init ----------------------
def init_ee(project_id=None, auto_create=False, project_prefix="ndvi-ee-proj", billing_account=None):
    def _try_initialize(proj):
        try:
            ee.Initialize(project=proj)
            print(f"✓ Earth Engine initialized with project: {proj}")
            return True
        except Exception:
            return False
    if project_id and _try_initialize(project_id): return
    env_proj = os.getenv("EE_PROJECT") or os.getenv("GOOGLE_CLOUD_PROJECT")
    if env_proj and _try_initialize(env_proj): return
    if _which("earthengine"):
        try:
            cli_proj = _run("earthengine get_project", check=False)
            if cli_proj and "No project set" not in cli_proj and _try_initialize(cli_proj): return
        except Exception: pass
    if _try_initialize("earthengine-legacy"):
        print("ⓘ Using legacy project 'earthengine-legacy'."); return
    try:
        print("🔐 Attempting Earth Engine authentication…"); ee.Authenticate()
    except Exception as e:
        print(f"Auth note: {e}")
    for candidate in [project_id, env_proj]:
        if candidate and _try_initialize(candidate): return
    if _which("earthengine"):
        try:
            cli_proj = _run("earthengine get_project", check=False)
            if cli_proj and "No project set" not in cli_proj and _try_initialize(cli_proj): return
        except Exception: pass
    if _try_initialize("earthengine-legacy"):
        print("ⓘ Using legacy project 'earthengine-legacy'."); return
    if not auto_create:
        raise RuntimeError("Earth Engine requires a Cloud Project. Pass --ee_project or set EE_PROJECT, "
                           "or run `earthengine set_project YOUR_PROJECT_ID`, or use --auto_create_project.")
    if not (_which("gcloud") and _which("earthengine")):
        raise RuntimeError("Auto-create needs both `gcloud` and `earthengine` CLIs.")
    suffix = uuid.uuid4().hex[:8]
    new_proj = f"{project_prefix}-{suffix}".lower()
    print(f"▶ Creating GCP project: {new_proj}")
    try: _run("gcloud auth login --brief", check=False)
    except Exception: pass
    _run(f"gcloud projects create {new_proj} --name={new_proj}", check=True)
    try:
        if billing_account:
            print("▶ Linking billing account …")
            _run(f"gcloud billing projects link {new_proj} --billing-account={billing_account}", check=True)
    except Exception as e:
        print(f"⚠️ Billing link failed or not permitted: {e}")
    print("▶ Enabling Earth Engine API …")
    _run(f"gcloud services enable earthengine.googleapis.com --project={new_proj}", check=True)
    print("▶ Setting Earth Engine project via CLI …")
    _run(f"earthengine set_project {new_proj}", check=True)
    if not _try_initialize(new_proj):
        raise RuntimeError("Failed to initialize Earth Engine with new project.")
    os.environ["EE_PROJECT"] = new_proj
    print(f"✓ Auto-created and initialized EE project: {new_proj}")

# ---------------------- Data acquisition ----------------------
def cdl_crop_mask(year, crop_name="CORN"):
    crop_map = {"CORN":1, "SOYBEANS":5, "WINTER WHEAT":24, "SPRING WHEAT":23, "COTTON":2, "ALFALFA":36,
                "RICE":3, "BARLEY":21, "SORGHUM":4}
    code = crop_map.get(crop_name.upper(), 1)
    ic = ee.ImageCollection("USDA/NASS/CDL")
    ic_year = ic.filter(ee.Filter.eq('year', year))
    ic_date = ic.filterDate(f"{year}-01-01", f"{year+1}-01-01")
    size_year = ic_year.size(); size_date = ic_date.size()
    img = ee.Image(ee.Algorithms.If(size_year.gt(0), ic_year.first(),
           ee.Algorithms.If(size_date.gt(0), ic_date.first(), ee.Image(1))))
    has_img = size_year.gt(0).Or(size_date.gt(0))
    mask = ee.Image(ee.Algorithms.If(has_img, img.select('cropland').eq(code), ee.Image(1)))
    return mask

def modis_vi_collection(year):
    start = ee.Date.fromYMD(year, 1, 1).advance(79, 'day')
    end   = ee.Date.fromYMD(year, 1, 1).advance(320, 'day')
    col = (ee.ImageCollection("MODIS/061/MOD13Q1")
           .filterDate(start, end)
           .select(['NDVI','EVI']))
    return col

def landsat_l2_sr_collection(year):
    start = ee.Date.fromYMD(year, 1, 1); end = ee.Date.fromYMD(year+1, 1, 1)
    l8 = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2").filterDate(start, end)
    l9 = ee.ImageCollection("LANDSAT/LC09/C02/T1_L2").filterDate(start, end)
    return l8.merge(l9)

def landsat_ndvi(img):
    red = img.select('SR_B4').multiply(0.0000275).add(-0.2)
    nir = img.select('SR_B5').multiply(0.0000275).add(-0.2)
    ndvi = nir.subtract(red).divide(nir.add(red)).rename('NDVI')
    return img.addBands(ndvi)

def landsat_cloudmask(img):
    qa = img.select('QA_PIXEL')
    cloud_shadow_bit = 1 << 3; clouds_bit = 1 << 4
    mask = qa.bitwiseAnd(cloud_shadow_bit).eq(0).And(qa.bitwiseAnd(clouds_bit).eq(0))
    return img.updateMask(mask)

def build_modis_series(geom, year, crop_name):
    mask = cdl_crop_mask(year, crop_name)
    col  = modis_vi_collection(year).map(lambda img: img.updateMask(mask)).sort('system:time_start')
    size = ee.Number(col.size()).getInfo()
    if size == 0: return []
    imgs = col.toList(col.size()); out = []
    for i in range(size):
        img = ee.Image(imgs.get(i)); dt = ee.Date(img.get('system:time_start'))
        doy = dt.getRelative('day', 'year').add(1).getInfo()
        stats = img.reduceRegion(reducer=ee.Reducer.mean(), geometry=geom, scale=250, maxPixels=1e13)
        nd = stats.get('NDVI').getInfo(); ev = stats.get('EVI').getInfo()
        ndvi = (float(nd)/1e4) if nd is not None else np.nan
        evi  = (float(ev)/1e4) if ev is not None else np.nan
        out.append({"year": int(year), "doy": int(doy), "ndvi": ndvi, "evi": evi, "source":"MODIS"})
    return [r for r in out if 80 <= r["doy"] <= 320]

def build_landsat_series(geom, year, crop_name, step_days=16):
    start = ee.Date.fromYMD(year,1,1).advance(79,'day')
    end   = ee.Date.fromYMD(year,1,1).advance(320,'day')
    col = (landsat_l2_sr_collection(year).map(landsat_cloudmask).map(landsat_ndvi))
    out = []; t = start
    while t.millis().lt(end.millis()).getInfo():
        t2 = t.advance(step_days, 'day'); win = col.filterDate(t, t2)
        if win.size().getInfo() > 0:
            med = win.median().select('NDVI')
            stats = med.reduceRegion(reducer=ee.Reducer.mean(), geometry=geom, scale=30, maxPixels=1e13)
            nd = stats.get('NDVI').getInfo()
            ndvi = float(nd) if nd is not None else np.nan
            doy = ee.Date(t.millis()).getRelative('day','year').add(1).getInfo()
            out.append({"year": int(year), "doy": int(doy), "ndvi": ndvi, "evi": np.nan, "source":"LANDSAT"})
        t = t2
    return [r for r in out if 80 <= r["doy"] <= 320]

def fetch_timeseries_for_county(state, county, crop, years, provider="MODIS"):
    geom, geoid = get_county_geom(state, county)
    records = []
    for y in years:
        if provider.upper() in ("LANDSAT","BOTH"): records += build_landsat_series(geom, y, crop, step_days=16)
        if provider.upper() in ("MODIS","BOTH"):   records += build_modis_series(geom, y, crop)
    for r in records:
        r.update({"state": state.upper(), "county": county, "geoid": geoid, "crop": crop.upper()})
    df = pd.DataFrame(records)
    if not df.empty:
        df = df[(df["doy"]>=80) & (df["doy"]<=320)].copy()
        df.sort_values(["geoid","year","doy","source"], inplace=True)
    return df

# ---------------------- Robust HTTP helper ----------------------
def _get_json_with_retries(url, params, max_retries=5, base_sleep=0.8):
    last_exc = None
    for i in range(max_retries):
        try:
            r = requests.get(url, params=params, timeout=60)
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(base_sleep * (2 ** i)); continue
            try: js = r.json()
            except Exception: return False, None, r.status_code, r.text
            return True, js, r.status_code, r.text
        except requests.exceptions.RequestException as e:
            last_exc = e; time.sleep(base_sleep * (2 ** i))
    return False, None, None, str(last_exc) if last_exc else ""

# ---------------------- NASS yields (robust) ----------------------
def nass_county_yield(state_abbr, county_name, crop_name, years, api_key=None):
    if api_key is None: api_key = os.environ.get("NASS_API_KEY", None)
    if api_key is None:
        raise RuntimeError("NASS_API_KEY not set. Get a free key at https://quickstats.nass.usda.gov/api")
    commodity = crop_name.upper(); url = "https://quickstats.nass.usda.gov/api/api_GET/"
    base = county_name.strip()
    variants = list(dict.fromkeys([base.title(), f"{base.title()} County", base.upper(), f"{base.upper()} COUNTY"]))
    dfs = []
    for y in years:
        got_value = False
        for v in variants:
            params = {"key": api_key, "source_desc": "SURVEY","sector_desc": "CROPS","group_desc": "FIELD CROPS",
                      "commodity_desc": commodity, "statisticcat_desc": "YIELD", "unit_desc": "BU / ACRE",
                      "agg_level_desc": "COUNTY", "state_alpha": state_abbr.upper(), "county_name": v,
                      "format": "JSON", "year": str(y)}
            ok, js, code, _ = _get_json_with_retries(url, params)
            if ok and isinstance(js, dict) and "error" in js and code == 400:
                params.pop("unit_desc", None); ok, js, code, _ = _get_json_with_retries(url, params)
            if not ok or code == 400 or not isinstance(js, dict): continue
            rows = js.get("data", [])
            if not rows: continue
            chosen = next((rr for rr in rows if rr.get("practice_desc","").upper()=="ALL PRODUCTION PRACTICES"), rows[0])
            val_s = chosen.get("Value", "").replace(",","").strip()
            try: yld = float(val_s)
            except: yld = np.nan
            dfs.append({"year": int(y), "y_true": yld}); got_value = True; break
        if not got_value:
            print(f"⚠️ NASS: no yield found for {county_name} ({variants[0]}) {state_abbr}, {y}. Filling NaN.")
            dfs.append({"year": int(y), "y_true": np.nan})
    return pd.DataFrame(dfs)

# ---------------------- Feature engineering ----------------------
def _to_numeric_safe(df, cols):
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def series_to_features(grp):
    g = grp.sort_values("doy").copy()
    doys = g["doy"].to_numpy()
    ndvi = pd.Series(g["ndvi"].to_numpy()).interpolate(limit_direction="both").fillna(method="bfill").fillna(method="ffill")
    evi  = pd.Series(g["evi"].to_numpy()).interpolate(limit_direction="both").fillna(method="bfill").fillna(method="ffill")
    ndvi = ndvi.fillna(ndvi.mean()).to_numpy()
    evi  = evi.fillna(evi.mean()).to_numpy()
    feats = {
        "AUC_ndvi": trapezoid_auc(ndvi, x_step=16),
        "peak_ndvi": float(np.nanmax(ndvi)) if len(ndvi)>0 else np.nan,
        "mean_ndvi": float(np.nanmean(ndvi)) if len(ndvi)>0 else np.nan,
        "tpeak_ndvi": timing_of_peak(ndvi, doys),
        "AUC_evi": trapezoid_auc(evi, x_step=16) if np.isfinite(evi).any() else np.nan,
        "peak_evi": float(np.nanmax(evi)) if np.isfinite(evi).any() else np.nan,
        "mean_evi": float(np.nanmean(evi)) if np.isfinite(evi).any() else np.nan,
        "tpeak_evi": timing_of_peak(evi, doys) if np.isfinite(evi).any() else np.nan,
    }
    seq_payload = {"doy": doys.tolist(),
                   "ndvi": [float(x) for x in ndvi],
                   "evi":  [float(x) for x in evi]}
    return feats, json.dumps(seq_payload)

def build_feature_table(ts_df):
    rows = []
    for (geoid, year, source), grp in ts_df.groupby(["geoid","year","source"]):
        feats, seq = series_to_features(grp)
        meta = grp.iloc[0][["state","county","crop"]].to_dict()
        rec = {"geoid":geoid, "year":int(year), "provider":source, **feats, "sequence":seq, **meta}
        rows.append(rec)
    df = pd.DataFrame(rows)
    cols = ["AUC_ndvi","peak_ndvi","mean_ndvi","tpeak_ndvi","AUC_evi","peak_evi","mean_evi","tpeak_evi"]
    df = _to_numeric_safe(df, cols)
    return df

# ---------------------- Provider helpers ----------------------
def provider_subdf(features_df, which):
    """
    Return provider-specific df with engineered features, sequence JSON, and y_true attached.
    For COMBINED: average tabular features across MODIS/LANDSAT, build combined sequence,
    and merge labels from original features_df.
    """
    keycols = ["geoid","county","state","crop","year"]

    if which in ("MODIS","LANDSAT"):
        sub = features_df[features_df["provider"]==which].copy()
        tabcols = ["AUC_ndvi","peak_ndvi","mean_ndvi","tpeak_ndvi",
                   "AUC_evi","peak_evi","mean_evi","tpeak_evi"]
        sub = _to_numeric_safe(sub, tabcols)
        return sub

    elif which == "COMBINED":
        dfM = features_df[features_df["provider"]=="MODIS"].copy()
        dfL = features_df[features_df["provider"]=="LANDSAT"].copy()
        tabcols = ["AUC_ndvi","peak_ndvi","mean_ndvi","tpeak_ndvi",
                   "AUC_evi","peak_evi","mean_evi","tpeak_evi","sequence"]
        m = dfM[keycols+tabcols].rename(columns={c:f"{c}_M" for c in tabcols})
        l = dfL[keycols+tabcols].rename(columns={c:f"{c}_L" for c in tabcols})
        df = m.merge(l, on=keycols, how="outer")

        out = pd.DataFrame()
        for col in keycols:
            out[col] = df[col]

        def _avg(a, b):
            a = pd.to_numeric(a, errors="coerce")
            b = pd.to_numeric(b, errors="coerce")
            return float(np.nanmean([a, b]))

        for col in ["AUC_ndvi","peak_ndvi","mean_ndvi","tpeak_ndvi",
                    "AUC_evi","peak_evi","mean_evi","tpeak_evi"]:
            out[col] = df[f"{col}_M"].combine(df[f"{col}_L"], func=_avg)

        combined_seq = []
        for _, row in df.iterrows():
            sM = row.get("sequence_M"); sL = row.get("sequence_L")
            dM = json.loads(sM) if isinstance(sM, str) else {"doy":[],"ndvi":[],"evi":[]}
            dL = json.loads(sL) if isinstance(sL, str) else {"doy":[],"ndvi":[],"evi":[]}
            doyM = np.array(dM.get("doy", []), dtype=float)
            ndM  = np.array(dM.get("ndvi",[]), dtype=float)
            evM  = np.array(dM.get("evi", []), dtype=float)
            doyL = np.array(dL.get("doy", []), dtype=float)
            ndL  = np.array(dL.get("ndvi",[]), dtype=float)

            if len(doyM)==0 and len(doyL)==0:
                combined_seq.append(json.dumps({"doy":[],"ndvi":[],"evi":[],"ndvi_ls":[]})); continue
            if len(doyM)==0:
                seq = {"doy": dL.tolist(), "ndvi": ndL.tolist(), "evi": [], "ndvi_ls": ndL.tolist()}
                combined_seq.append(json.dumps(seq)); continue
            if len(doyL) > 1:
                ndL_interp = np.interp(doyM, doyL, ndL, left=ndL[0], right=ndL[-1])
            elif len(doyL) == 1:
                ndL_interp = np.full_like(doyM, fill_value=ndL[0], dtype=float)
            else:
                ndL_interp = np.full_like(doyM, fill_value=np.nan, dtype=float)

            seq = {"doy": doyM.tolist(),
                   "ndvi": ndM.tolist(),
                   "evi":  evM.tolist(),
                   "ndvi_ls": ndL_interp.tolist()}
            combined_seq.append(json.dumps(seq))

        out["sequence"] = combined_seq
        out["provider"] = "COMBINED"

        # Bring back labels (y_true) from original dataframe
        labels = (features_df
                  .dropna(subset=["y_true"])
                  .sort_values(keycols)
                  .groupby(keycols, as_index=False)["y_true"]
                  .first())
        out = out.merge(labels, on=keycols, how="left")

        out = _to_numeric_safe(out, ["AUC_ndvi","peak_ndvi","mean_ndvi","tpeak_ndvi",
                                     "AUC_evi","peak_evi","mean_evi","tpeak_evi"])
        return out

    else:
        raise ValueError("Unknown provider subset")

# ---------------------- Sequence tensors ----------------------
def make_lstm_dataset_provider(df, window_len=None):
    seqs = []
    for s, prov in zip(df["sequence"].tolist(), df["provider"].tolist()):
        d = json.loads(s) if isinstance(s, str) else {"doy":[],"ndvi":[],"evi":[]}
        nd = np.array(d.get("ndvi", []), dtype=float)
        ev = np.array(d.get("evi",  []), dtype=float)
        doy = np.array(d.get("doy",  []), dtype=float)
        if len(doy)==0:
            seqs.append(np.zeros((0,1), dtype=np.float32)); continue
        # add seasonal encoding
        sin_doy = np.sin(2.0 * math.pi * (doy / 365.0))
        cos_doy = np.cos(2.0 * math.pi * (doy / 365.0))
        chans = [nd]
        if prov in ("MODIS","COMBINED") and len(ev)>0 and np.isfinite(ev).any():
            chans.append(ev)
        chans.append(sin_doy); chans.append(cos_doy)
        Tm = min(*[len(c) for c in chans])
        X = np.stack([c[:Tm] for c in chans], axis=-1)
        seqs.append(X.astype(np.float32))
    if len(seqs)==0:
        return np.zeros((0,1,1), dtype=np.float32)
    T = max(arr.shape[0] for arr in seqs) if window_len is None else window_len
    C = max(arr.shape[1] for arr in seqs)
    X = np.zeros((len(seqs), T, C), dtype=np.float32)
    for i, arr in enumerate(seqs):
        if arr.shape[0]==0: continue
        t = min(T, arr.shape[0])
        X[i, :t, :arr.shape[1]] = arr[:t, :]
        if t < T: X[i, t:, :arr.shape[1]] = arr[t-1:t, :]
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True) + 1e-6
    return (X - mu)/sd

# ---------------------- LSTM model (BiLSTM + Attention) ----------------------
def build_lstm(input_timesteps, channels, width=64, l2=1e-4, dr=0.2):
    inputs = layers.Input(shape=(input_timesteps, channels))
    x = layers.GaussianNoise(0.02)(inputs)
    x = layers.SpatialDropout1D(0.15)(x)
    x = layers.Bidirectional(layers.LSTM(
        width, return_sequences=True,
        kernel_regularizer=regularizers.l2(l2),
        recurrent_regularizer=regularizers.l2(l2)
    ))(x)
    attn = layers.MultiHeadAttention(num_heads=2, key_dim=max(8, channels))(x, x)
    x = layers.Add()([x, attn]); x = layers.LayerNormalization()(x)
    x = layers.Concatenate()([layers.GlobalAveragePooling1D()(x), layers.GlobalMaxPooling1D()(x)])
    x = layers.Dense(width, activation="relu", kernel_regularizer=regularizers.l2(l2))(x)
    x = layers.Dropout(dr)(x)
    out = layers.Dense(1)(x)
    model = models.Model(inputs, out)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse",
                  metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae")])
    return model

def r2_score(y, yhat):
    y = np.asarray(y).reshape(-1); yhat = np.asarray(yhat).reshape(-1)
    ss_res = np.sum((y - yhat)**2); ss_tot = np.sum((y - y.mean())**2)
    return float(1 - ss_res/ss_tot) if ss_tot > 0 else np.nan

# ---------------------- Tabular models + robust imputation ----------------------
TAB_COLS = ["AUC_ndvi","peak_ndvi","mean_ndvi","tpeak_ndvi","AUC_evi","peak_evi","mean_evi","tpeak_evi"]

def _provider_fill_specials(dfX):
    dfX = _to_numeric_safe(dfX, TAB_COLS)
    if "provider" in dfX.columns:
        mask_ls = (dfX["provider"]=="LANDSAT")
        evi_cols = ["AUC_evi","peak_evi","mean_evi","tpeak_evi"]
        dfX.loc[mask_ls, evi_cols] = dfX.loc[mask_ls, evi_cols].fillna(0.0)
    return dfX

def make_tabular_train_matrix(df_train):
    Xtr = df_train[TAB_COLS + (["provider"] if "provider" in df_train.columns else [])].copy()
    Xtr = _provider_fill_specials(Xtr)
    Xtr_only = Xtr[TAB_COLS]
    imputer = SimpleImputer(strategy="median")
    Xtr_imp = imputer.fit_transform(Xtr_only)
    Xtr_imp = np.nan_to_num(Xtr_imp, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    ytr = df_train["y_true"].to_numpy().astype(np.float32)
    return Xtr_imp, ytr, imputer

def make_tabular_test_matrix(df_test, imputer):
    Xte = df_test[TAB_COLS + (["provider"] if "provider" in df_test.columns else [])].copy()
    Xte = _provider_fill_specials(Xte)
    Xte_only = Xte[TAB_COLS]
    Xte_imp = imputer.transform(Xte_only)
    Xte_imp = np.nan_to_num(Xte_imp, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    yte = df_test["y_true"].to_numpy().astype(np.float32)
    return Xte_imp, yte

def fit_tabular_models(Xtr, ytr):
    if not np.isfinite(Xtr).all():
        Xtr = np.nan_to_num(Xtr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    lin = LinearRegression()
    lin.fit(Xtr, ytr)
    rf = RandomForestRegressor(n_estimators=400, max_depth=None, random_state=42, n_jobs=-1)
    rf.fit(Xtr, ytr)
    mean_pred = float(np.mean(ytr))
    return lin, rf, mean_pred

def predict_tabular(models, X):
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    lin, rf, mean_pred = models
    return {
        "BaselineMean": np.full((len(X),), mean_pred, dtype=float),
        "Linear": lin.predict(X),
        "RandomForest": rf.predict(X)
    }

# ---------------------- Parallel-safe wrappers ----------------------
_gee_semaphore = threading.Semaphore(6)
def _fetch_cty_worker(args):
    state, county, crop, years, provider = args
    try:
        with _gee_semaphore:
            df_cty = fetch_timeseries_for_county(state, county, crop, years, provider=provider)
        counts = df_cty.groupby("source").size().to_dict() if not df_cty.empty else {}
        return (county, df_cty, counts, None)
    except Exception as e:
        return (county, pd.DataFrame(), {}, str(e))

def _yield_cty_worker(args):
    state, county, crop, years = args
    try:
        y = nass_county_yield(state, county, crop, years)
        y["county"] = county
        return (county, y, None)
    except Exception as e:
        return (county, pd.DataFrame(), str(e))

# ---------------------- CV & Training ----------------------
def make_lstm_dataset(df, window_len=None):
    # (legacy hook if ever needed)
    return make_lstm_dataset_provider(df, window_len=window_len)

def loyo_cv(features_df, label_col="y_true", seq_col="sequence", epochs=150, verbose=False, provider_choice="MODIS"):
    rows = []
    df = provider_subdf(features_df, provider_choice)
    df = df.dropna(subset=[label_col]).copy()
    if df.empty:
        return pd.DataFrame(columns=["provider","year","model","MAE","RMSE","R2","n"])
    all_years = sorted(df["year"].unique())
    X_seq_all = make_lstm_dataset_provider(df, window_len=None)
    T, C = X_seq_all.shape[1], X_seq_all.shape[2]
    y_all = df[label_col].to_numpy().astype(np.float32)
    for test_year in all_years:
        tr_idx = df["year"] != test_year
        te_idx = df["year"] == test_year
        if te_idx.sum() == 0 or tr_idx.sum() < 5: continue
        df_tr, df_te = df[tr_idx], df[te_idx]
        # Tabular with imputation
        Xtr_tab, ytr, imputer = make_tabular_train_matrix(df_tr)
        Xte_tab, yte          = make_tabular_test_matrix(df_te, imputer)
        if not np.isfinite(Xtr_tab).all(): Xtr_tab = np.nan_to_num(Xtr_tab, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        if not np.isfinite(Xte_tab).all(): Xte_tab = np.nan_to_num(Xte_tab, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        tab_models = fit_tabular_models(Xtr_tab, ytr)
        tab_preds  = predict_tabular(tab_models, Xte_tab)
        # LSTM
        Xtr_seq = X_seq_all[tr_idx]; Xte_seq = X_seq_all[te_idx]
        seq_model = build_lstm(input_timesteps=T, channels=C, width=64, l2=1e-4, dr=0.2)
        early = tf.keras.callbacks.EarlyStopping(monitor="val_mae", patience=15, min_delta=0.1, restore_best_weights=True)
        plateau = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_mae", factor=0.5, patience=5, min_lr=1e-5, verbose=0)
        seq_model.fit(Xtr_seq, ytr, validation_split=0.25, epochs=epochs, verbose=0,
                      callbacks=[early, plateau], batch_size=8)
        lstm_pred = seq_model.predict(Xte_seq, verbose=0).reshape(-1)
        tf.keras.backend.clear_session()
        model_preds = {"BaselineMean": tab_preds["BaselineMean"], "Linear": tab_preds["Linear"],
                       "RandomForest": tab_preds["RandomForest"], "LSTM": lstm_pred}
        for mname, yhat in model_preds.items():
            mae = float(np.mean(np.abs(yte - yhat)))
            rmse = float(np.sqrt(np.mean((yte - yhat)**2)))
            r2 = r2_score(yte, yhat)
            rows.append({"provider":provider_choice, "year": int(test_year),
                         "model": mname, "MAE": mae, "RMSE": rmse, "R2": r2, "n": int(len(yte))})
        if verbose: print(f"[{provider_choice}] LOYO year {test_year}: done.")
    return pd.DataFrame(rows)

# ---------------------- Main ----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required=True, help="State abbreviation, e.g., IA")
    ap.add_argument("--counties", help='Semicolon-separated list, e.g., "Story;Polk;Boone"')
    ap.add_argument("--all_counties", action="store_true", help="Process every county in the state (from TIGER).")
    ap.add_argument("--exclude_counties", default="", help='With --all_counties, skip list, e.g., "Polk;Story"')
    ap.add_argument("--crop", default="CORN", help="Crop (NASS/CDL): CORN, SOYBEANS, WINTER WHEAT, etc.")
    ap.add_argument("--start_year", type=int, default=2003)
    ap.add_argument("--end_year", type=int, default=2023)
    ap.add_argument("--provider", choices=["MODIS","LANDSAT","BOTH"], default="MODIS",
                    help="MODIS, LANDSAT, or BOTH (adds COMBINED).")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_workers", type=int, default=max(4, os.cpu_count() or 4),
                    help="Thread pool size for county & NASS parallelism.")
    # EE project controls
    ap.add_argument("--ee_project", default=os.getenv("EE_PROJECT"),
                    help="Explicit Earth Engine Cloud Project ID (or set EE_PROJECT env).")
    ap.add_argument("--auto_create_project", action="store_true",
                    help="Auto-create a new GCP project if none is available.")
    ap.add_argument("--project_prefix", default="ndvi-ee-proj",
                    help="Prefix for auto-created GCP project IDs.")
    ap.add_argument("--billing_account", default=os.getenv("BILLING_ACCOUNT"),
                    help="Optional billing account ID to link when auto-creating.")
    args = ap.parse_args()

    np.random.seed(args.seed); tf.random.set_seed(args.seed); ensure_dir(OUT_DIR)

    # Initialize EE
    init_ee(project_id=args.ee_project, auto_create=args.auto_create_project,
            project_prefix=args.project_prefix, billing_account=args.billing_account)

    years = list(range(args.start_year, args.end_year+1))
    # Determine counties
    if args.all_counties:
        all_ctys = list_state_counties(args.state)
        excludes = [c.strip() for c in args.exclude_counties.split(";") if c.strip()]
        counties = [c for c in all_ctys if c not in set(excludes)]
        print(f"▶ Using ALL counties in {args.state} ({len(counties)} total){' minus '+str(excludes) if excludes else ''}.")
    else:
        if not args.counties: raise SystemExit("Provide --counties \"A;B;C\" or use --all_counties")
        counties = [c.strip() for c in args.counties.split(";") if c.strip()]
        print(f"▶ Using provided counties ({len(counties)}): {counties}")

    print(f"▶ Downloading VI for {len(counties)} county(ies) in {args.state}, crop={args.crop}, years={years} [provider={args.provider}]")
    provider_for_fetch = args.provider if args.provider!="BOTH" else "BOTH"

    # 1) Time series (PARALLEL)
    ts_all = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        futs = {ex.submit(_fetch_cty_worker, (args.state, cty, args.crop, years, provider_for_fetch)): cty
                for cty in counties}
        done_ct = 0
        for fut in as_completed(futs):
            cty = futs[fut]
            county, df_cty, counts, err = fut.result(); done_ct += 1
            if err:
                print(f"  [{done_ct:>3}/{len(counties)}] {county:<18} ✖ {err}"); continue
            if df_cty is None or df_cty.empty:
                print(f"  [{done_ct:>3}/{len(counties)}] {county:<18} ⚠️ no time series"); continue
            ts_all.append(df_cty)
            counts_str = ", ".join([f"{k}:{v}" for k,v in counts.items()])
            print(f"  [{done_ct:>3}/{len(counties)}] {county:<18} → {counts_str}")

    if len(ts_all) == 0:
        raise RuntimeError("No time series were returned; check inputs and provider.")
    ts_df = pd.concat(ts_all, ignore_index=True)
    ts_df.to_csv(TS_CSV, index=False)
    print(f"✓ Saved time series: {TS_CSV} ({len(ts_df)} rows)")

    # 2) Features (per county×year×provider)
    feat_df = build_feature_table(ts_df)
    feat_df.to_csv(FEAT_CSV, index=False)
    print(f"✓ Saved features: {FEAT_CSV} ({len(feat_df)} county-year-provider rows)")

    # 3) NASS yields (PARALLEL)
    print("▶ Fetching USDA NASS county yields (parallel) …")
    yld_parts = []
    with ThreadPoolExecutor(max_workers=min(12, args.max_workers)) as ex:
        futs = {ex.submit(_yield_cty_worker, (args.state, cty, args.crop, years)): cty for cty in counties}
        done_ct = 0
        for fut in as_completed(futs):
            cty = futs[fut]
            county, y_df, err = fut.result(); done_ct += 1
            if err:
                print(f"  [{done_ct:>3}/{len(counties)}] {county:<18} ✖ {err}"); continue
            if y_df.empty:
                print(f"  [{done_ct:>3}/{len(counties)}] {county:<18} ⚠️ no yield data")
            yld_parts.append(y_df); time.sleep(0.05)

    yld_df = pd.concat(yld_parts, ignore_index=True) if len(yld_parts)>0 else pd.DataFrame(columns=["county","year","y_true"])
    geo_map = (feat_df.groupby(["county"])["geoid"].agg(lambda x: x.iloc[0]).reset_index())
    df = feat_df.merge(yld_df, on=["county","year"], how="left")
    df = df.merge(geo_map, on="county", how="left", suffixes=("","_geo"))
    df["geoid"] = df["geoid_geo"].fillna(df["geoid"]); df.drop(columns=["geoid_geo"], inplace=True)
    n_lab = df["y_true"].notna().sum()
    print(f"✓ Yield labels found: {n_lab}/{len(df)} county-year-provider rows")

    # Providers to run
    run_providers = ["MODIS","LANDSAT"] if args.provider != "BOTH" else ["MODIS","LANDSAT","COMBINED"]

    # 4) LOYO CV per provider
    cv_all = []
    for prov in run_providers:
        cv = loyo_cv(df, label_col="y_true", seq_col="sequence", epochs=args.epochs, verbose=False, provider_choice=prov)
        cv_all.append(cv)
    cv = pd.concat(cv_all, ignore_index=True) if len(cv_all)>0 else pd.DataFrame(columns=["provider","year","model","MAE","RMSE","R2","n"])
    cv.to_csv(CV_CSV, index=False)
    print(f"✓ Saved LOYO CV metrics: {CV_CSV}")

    # 5) Final test on the last year (per provider)
    test_year = max(y for y in years if y in set(df.dropna(subset=['y_true'])['year'].unique())) if n_lab > 0 else None
    final_preds = []; comp_rows = []

    if test_year is not None:
        for prov in run_providers:
            sub = provider_subdf(df, prov)
            test_rows = sub[sub["year"]==test_year].dropna(subset=["y_true"]).copy()
            train_rows = sub[sub["year"]!=test_year].dropna(subset=["y_true"]).copy()
            if len(test_rows)==0 or len(train_rows)<5:
                print(f"⚠️ [{prov}] Not enough labeled data for final test."); continue

            # Tabular with imputation
            Xtr, ytr, imp = make_tabular_train_matrix(train_rows)
            Xte, yte = make_tabular_test_matrix(test_rows, imp)
            if not np.isfinite(Xtr).all(): Xtr = np.nan_to_num(Xtr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
            if not np.isfinite(Xte).all(): Xte = np.nan_to_num(Xte, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
            tab_models = fit_tabular_models(Xtr, ytr)
            tab_preds = predict_tabular(tab_models, Xte)

            # LSTM
            Xseq_tr = make_lstm_dataset_provider(train_rows, window_len=None)
            T, C = Xseq_tr.shape[1], Xseq_tr.shape[2]
            Xseq_te = make_lstm_dataset_provider(test_rows, window_len=T)
            seq_model = build_lstm(input_timesteps=T, channels=C, width=64, l2=1e-4, dr=0.2)
            early = tf.keras.callbacks.EarlyStopping(monitor="val_mae", patience=15, min_delta=0.1, restore_best_weights=True)
            plateau = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_mae", factor=0.5, patience=5, min_lr=1e-5, verbose=0)
            seq_model.fit(Xseq_tr, ytr, validation_split=0.25, epochs=args.epochs, verbose=0,
                          callbacks=[early, plateau], batch_size=8)
            lstm_pred = seq_model.predict(Xseq_te, verbose=0).reshape(-1)
            tf.keras.backend.clear_session()

            preds_map = {"BaselineMean": tab_preds["BaselineMean"], "Linear": tab_preds["Linear"],
                         "RandomForest": tab_preds["RandomForest"], "LSTM": lstm_pred}
            for i, (_, row) in enumerate(test_rows.iterrows()):
                for mname, arr in preds_map.items():
                    final_preds.append({"provider": prov, "state": row["state"], "county": row["county"], "geoid": row["geoid"],
                                        "year": int(test_year), "model": mname,
                                        "y_true": float(row["y_true"]), "y_pred": float(arr[i]),
                                        "resid": float(arr[i] - row["y_true"])})
            for mname, arr in preds_map.items():
                mae = float(np.mean(np.abs(yte - arr)))
                rmse = float(np.sqrt(np.mean((yte - arr)**2)))
                r2 = r2_score(yte, arr)
                comp_rows.append({"provider": prov, "model": mname,
                                  "CV_R2": np.nan, "CV_MAE": np.nan, "CV_RMSE": np.nan,
                                  "FinalTestYear": int(test_year), "FinalTest_R2": r2,
                                  "FinalTest_MAE": mae, "FinalTest_RMSE": rmse})

    if len(final_preds) > 0:
        pd.DataFrame(final_preds).to_csv(TEST_PRED_CSV, index=False)
        print(f"✓ Saved final-year predictions: {TEST_PRED_CSV}")

    # 6) Model comparison table
    if not cv.empty:
        cv_summary = (cv.groupby(["provider","model"])
                        .agg(CV_R2=("R2","mean"), CV_MAE=("MAE","mean"), CV_RMSE=("RMSE","mean"))
                        .reset_index())
    else:
        cv_summary = pd.DataFrame(columns=["provider","model","CV_R2","CV_MAE","CV_RMSE"])
    comp_test = pd.DataFrame(comp_rows) if len(comp_rows)>0 else pd.DataFrame(
        columns=["provider","model","CV_R2","CV_MAE","CV_RMSE","FinalTestYear","FinalTest_R2","FinalTest_MAE","FinalTest_RMSE"])
    comp_df = cv_summary.merge(comp_test, on=["provider","model"], how="outer")
    comp_df.to_csv(COMPARE_CSV, index=False)
    print(f"✓ Saved model comparison: {COMPARE_CSV}")

    print("\nAll done. Key outputs:")
    print(" -", TS_CSV)
    print(" -", FEAT_CSV)
    print(" -", CV_CSV)
    if os.path.exists(TEST_PRED_CSV): print(" -", TEST_PRED_CSV)
    print(" -", COMPARE_CSV)

if __name__ == "__main__":
    main()
