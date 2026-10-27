# dashboard_streamlit.py
"""
Streamlit app untuk visualisasi radiasi:
- load CSV sample atau upload file
- pilih kolom radiasi
- resample per-menit menjadi per-jam (mean)
- plot time-series interaktif (gabungan parameter)
- profil radiasi rata-rata per-jam untuk rentang tanggal (06:00 - 18:00)
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO

st.set_page_config(page_title="Radiation Dashboard", layout="wide")

# -------------------------
# Helpers & cached loaders
# -------------------------
@st.cache_data
def load_csv(path_or_buffer):
    # Try several common datetime column parsing strategies.
    # Return DataFrame with a parsed datetime index.
    # path_or_buffer can be a path string or a file-like object.
    df = pd.read_csv(path_or_buffer)
    # try to find a datetime-like column
    datetime_cols = [c for c in df.columns if 'time' in c.lower() or 'date' in c.lower()]
    if not datetime_cols:
        # fallback: assume first column is datetime
        datetime_cols = [df.columns[0]]
    # try parsing
    last_exc = None
    for col in datetime_cols:
        try:
            # Some CSVs have day/month/year single-string like "9/12/2025 16:54"
            df[col] = pd.to_datetime(df[col], dayfirst=False, errors='coerce')
            if df[col].isna().all():
                # try dayfirst=True
                df[col] = pd.to_datetime(df[col].astype(str), dayfirst=True, errors='coerce')
            if df[col].notna().any():
                df = df.set_index(col)
                df.index.name = 'datetime'
                return df
        except Exception as e:
            last_exc = e
    # If reach here, try to coerce the first column anyway
    try:
        df[df.columns[0]] = pd.to_datetime(df[df.columns[0]].astype(str), errors='coerce', dayfirst=True)
        df = df.set_index(df.columns[0])
        df.index.name = 'datetime'
        return df
    except Exception as e:
        raise ValueError("Gagal mem-parse kolom datetime. Coba unggah CSV dengan kolom datetime yang jelas.") from e

@st.cache_data
def resample_to_hourly(df):
    # Ensure index is tz-naive datetime64
    df2 = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df2.index):
        raise ValueError("Index bukan datetime.")
    # If there are multiple numeric columns, compute mean per hour
    hourly = df2.resample('h').mean()
    return hourly

def compute_hourly_profile(hourly_df, start_date, end_date, selected_cols, hour_min=6, hour_max=18):
    # Filter by date range (based on hourly index)
    mask = (hourly_df.index.date >= pd.to_datetime(start_date).date()) & (hourly_df.index.date <= pd.to_datetime(end_date).date())
    df_range = hourly_df.loc[mask]
    if df_range.empty:
        return pd.DataFrame()  # empty
    # create columns for hour and date
    df2 = df_range.copy()
    df2['hour'] = df2.index.hour
    df2['date'] = df2.index.date
    # restrict to hour_min..hour_max inclusive
    df2 = df2[(df2['hour'] >= hour_min) & (df2['hour'] <= hour_max)]
    # group by hour and average across dates
    profile = df2.groupby('hour')[selected_cols].mean()
    # ensure hours present
    all_hours = pd.Index(range(hour_min, hour_max+1), name='hour')
    profile = profile.reindex(all_hours, fill_value=np.nan)
    return profile

def to_bytes_download(df, filename="data.csv"):
    b = BytesIO()
    df.to_csv(b)
    b.seek(0)
    return b

# -------------------------
# UI layout
# -------------------------
st.title("Radiation Dashboard — Interactive")
st.markdown(
    """
    Instruksi singkat:
    - App akan coba memuat `/mnt/data/asrs-bsrn.csv` (sample) otomatis jika tersedia.
    - Kamu bisa unggah CSV lain lewat uploader.
    - Pilih kolom radiasi, lalu pilih rentang tanggal untuk profil jam.
    """
)

col1, col2 = st.columns([1,3])

with col1:
    st.header("Data input")
    use_sample = False
    # Try default sample file path
    default_path = "/mnt/data/asrs-bsrn.csv"
    try:
        with open(default_path, 'rb'):
            use_sample = True
    except Exception:
        use_sample = False

    uploaded_file = st.file_uploader("Upload CSV (opsional). Jika kosong, app akan coba pakai sample.", type=["csv"])
    df = None
    try:
        if uploaded_file is not None:
            df = load_csv(uploaded_file)
            st.success("CSV uploaded & parsed.")
        else:
            if use_sample:
                df = load_csv(default_path)
                st.info(f"Memuat sample CSV dari `{default_path}`.")
            else:
                st.warning("Tidak ada sample CSV di server. Unggah file CSV untuk melanjutkan.")
    except Exception as e:
        st.error(f"Gagal memuat CSV: {e}")

    if df is None:
        st.stop()

    st.write("Preview data (index = datetime):")
    st.dataframe(df.head(6))

with col2:
    st.header("Pilihan & Visualisasi")
    # find candidate radiation columns
    candidate_cols = [c for c in df.columns if any(k in c.lower() for k in ['dni', 'diffuse', 'global', 'reflected', 'rad'])]
    # ensure unique
    candidate_cols = list(dict.fromkeys(candidate_cols + list(df.columns)))  # keep all columns if radiation names not found
    st.caption("Kolom yang terdeteksi (pilih yang ingin dipakai):")
    selected_cols = st.multiselect("Pilih kolom radiasi (minimal 1)", options=candidate_cols,
                                   default=[c for c in candidate_cols if c in ['DNI_rad_Avg','diffuse_rad_Avg','global_rad_Avg','reflected_rad_Avg']][:4])
    if not selected_cols:
        st.warning("Pilih minimal 1 kolom untuk melanjutkan.")
        st.stop()

    # Resample to hourly
    try:
        hourly = resample_to_hourly(df[selected_cols])
    except Exception as e:
        st.error(f"Gagal resample ke hourly: {e}")
        st.stop()

    st.markdown("### Time-series (hourly average)")
    # time range selector
    min_dt = hourly.index.min()
    max_dt = hourly.index.max()
    if pd.isna(min_dt) or pd.isna(max_dt):
        st.error("Index datetime kosong setelah resample.")
        st.stop()

    date_range = st.date_input("Pilih rentang tanggal untuk plot time-series",
                               value=(min_dt.date(), max_dt.date()),
                               min_value=min_dt.date(),
                               max_value=max_dt.date())

    # Filter hourly by date range
    start_date, end_date = date_range
    mask = (hourly.index.date >= pd.to_datetime(start_date).date()) & (hourly.index.date <= pd.to_datetime(end_date).date())
    hourly_filtered = hourly.loc[mask]

    if hourly_filtered.empty:
        st.warning("Tidak ada data dalam rentang tanggal tersebut.")
    else:
        # plot combined lines with plotly
        fig = go.Figure()
        for col in selected_cols:
            fig.add_trace(go.Scatter(x=hourly_filtered.index, y=hourly_filtered[col],
                                     mode='lines+markers', name=col,
                                     hovertemplate='%{x|%Y-%m-%d %H:%M}: %{y:.2f}<extra></extra>'))
        fig.update_layout(
            xaxis_title="Datetime (hourly)",
            yaxis_title="Radiation (avg per hour)",
            hovermode="x unified",
            legend_title="Parameter"
        )
        st.plotly_chart(fig, use_container_width=True)

    # Download hourly CSV
    st.download_button("Download hourly-averaged CSV (filtered)", data=to_bytes_download(hourly_filtered),
                       file_name="hourly_averaged.csv", mime="text/csv")

    # -------------------------
    # Hourly profile across dates
    # -------------------------
    st.markdown("---")
    st.subheader("Hourly profile across dates (rata-rata per jam)")
    # default profile date range: user requested 12 - 25 Sep (auto choose if within range)
    default_start = pd.to_datetime("2025-09-12").date()
    default_end = pd.to_datetime("2025-09-25").date()
    # clamp defaults to data min/max
    ds = max(min_dt.date(), default_start)
    de = min(max_dt.date(), default_end)
    profile_range = st.date_input("Pilih rentang tanggal untuk profile (multi-day average)", value=(ds, de),
                                  min_value=min_dt.date(), max_value=max_dt.date())

    prof_start, prof_end = profile_range
    profile = compute_hourly_profile(hourly, prof_start, prof_end, selected_cols, hour_min=6, hour_max=18)
    if profile.empty:
        st.warning("Tidak ada data untuk membuat profile per-jam di rentang tanggal ini.")
    else:
        # profile is DataFrame indexed by hour
        profile_reset = profile.reset_index().melt(id_vars='hour', var_name='parameter', value_name='value')
        fig2 = px.line(profile_reset, x='hour', y='value', color='parameter',
                       markers=True,
                       labels={'hour': 'Hour of day', 'value': 'Average radiation', 'parameter': 'Parameter'})
        fig2.update_xaxes(tickmode='array', tickvals=list(range(6, 19)))
        fig2.update_layout(title=f"Hourly profile mean per-hour ({prof_start} → {prof_end})",
                           xaxis_title="Hour (06:00 - 18:00)", yaxis_title="Average radiation")
        st.plotly_chart(fig2, use_container_width=True)

        st.markdown("Rata-rata nilai per jam (tabel):")
        st.dataframe(profile)

        st.download_button("Download hourly-profile CSV", data=to_bytes_download(profile),
                           file_name="hourly_profile.csv", mime="text/csv")

st.markdown("---")
st.caption("Catatan: app mengasumsikan jam pada CSV adalah lokal (WIB). Jika ingin melakukan konversi timezone, lakukan pra-proses pada file CSV atau modifikasi kode ini.")
