import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import pydeck as pdk
import pickle
import joblib
import warnings
warnings.filterwarnings('ignore')

# Import ML libraries that might be needed for unpickling
try:
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.pipeline import Pipeline
    from xgboost import XGBClassifier
    import xgboost
except ImportError as e:
    st.warning(f"⚠️ Missing ML library: {e}. Some features may not work.")

# =========================================================
# 0. Setup & Helper Functions
# =========================================================

# ตั้งค่าหน้าเว็บ (ต้องบรรทัดแรกสุดของ Streamlit command)
st.set_page_config(page_title="Traffy Fondue Explorer (Auto Map)", layout="wide")

@st.cache_resource
def load_model():
    """โหลดไฟล์ Model (.pkl)"""
    try:
        # ตรวจสอบชื่อไฟล์โมเดลของคุณให้ถูกต้อง
        # Use joblib instead of pickle for better compatibility with sklearn/xgboost objects
        return joblib.load('traffy_model_weather.pkl')
    except FileNotFoundError:
        st.warning("⚠️ ไม่พบไฟล์ 'traffy_model_weather.pkl' - ตรวจสอบให้แน่ใจว่าไฟล์อยู่ในไดเรกทอรี่เดียวกับ script นี้")
        return None
    except (pickle.UnpicklingError, ModuleNotFoundError, AttributeError) as e:
        st.error(f"❌ เกิดข้อผิดพลาดในการโหลดโมเดล: {type(e).__name__}: {e}")
        st.info("💡 ความช่วยเหลือ: โปรดติดตั้งไลบรารีต่อไปนี้:")
        st.code("pip install scikit-learn xgboost pandas numpy joblib", language="bash")
        return None
    except Exception as e:
        st.error(f"❌ เกิดข้อผิดพลาดที่ไม่คาดคิด: {type(e).__name__}: {e}")
        return None

def preprocess_for_prediction(df, model_pkg):
    """
    [IMPORTANT] ฟังก์ชันการเตรียมข้อมูลสำหรับการทำนาย
    เตรียมข้อมูลให้ตรงกับรูปแบบที่โมเดล XGBoost ต้องการ
    รวมการ Encode คอลัมน์เชิงหมวดหมู่ และสร้าง Time Features
    """
    from sklearn.preprocessing import LabelEncoder
    
    # สำเนา DataFrame เพื่อไม่แก้ไขข้อมูลต้นฉบับ
    df_processed = df.copy()
    
    # ====== Step 1: Create Time Features from Timestamp ======
    if 'timestamp' in df_processed.columns:
        try:
            df_processed['timestamp'] = pd.to_datetime(df_processed['timestamp'], errors='coerce')
            df_processed['hour'] = df_processed['timestamp'].dt.hour
            df_processed['day_of_week'] = df_processed['timestamp'].dt.dayofweek
            df_processed['month'] = df_processed['timestamp'].dt.month
        except Exception as e:
            st.warning(f"⚠️ ไม่สามารถแตก timestamp: {e}")
    
    # ====== Step 2: Create Text Features ======
    if 'comment' in df_processed.columns:
        df_processed['comment_len'] = df_processed['comment'].astype(str).apply(len)
    else:
        df_processed['comment_len'] = 0
    
    # ====== Step 3: Encode Categorical Columns ======
    cols_to_encode = ['district', 'subdistrict', 'type 1']
    org_col = 'organization_1' if 'organization_1' in df_processed.columns else 'organization'
    if org_col in df_processed.columns:
        cols_to_encode.append(org_col)
    
    # ใช้ Encoders จากโมเดล หรือสร้างใหม่
    if model_pkg and 'encoders' in model_pkg:
        encoders_dict = model_pkg['encoders']
    else:
        encoders_dict = {}
    
    for col in cols_to_encode:
        if col in df_processed.columns:
            # เติม Unknown สำหรับค่าว่าง
            df_processed[col] = df_processed[col].fillna('Unknown').astype(str)
            
            # ถ้ามี Encoder จากโมเดล ให้ใช้นั้น
            if col in encoders_dict:
                try:
                    # Handle unknown categories (from prediction data)
                    le = encoders_dict[col]
                    # ถ้าค่าไม่อยู่ใน encoder ให้ assign 0
                    df_processed[f'{col}_enc'] = df_processed[col].map(
                        lambda x: le.transform([x])[0] if x in le.classes_ else 0
                    )
                except Exception as e:
                    st.warning(f"⚠️ ไม่สามารถ encode {col}: {e}")
                    df_processed[f'{col}_enc'] = 0
            else:
                # สร้าง Encoder ใหม่ (ถ้าไม่มีจากโมเดล)
                le = LabelEncoder()
                try:
                    df_processed[f'{col}_enc'] = le.fit_transform(df_processed[col])
                except Exception as e:
                    st.warning(f"⚠️ ไม่สามารถ encode {col}: {e}")
                    df_processed[f'{col}_enc'] = 0
    
    # ====== Step 4: Select Only Required Features ======
    # รายชื่อ Feature ที่โมเดลคาดหวัง (จากการ Train)
    required_features = [
        'district_enc', 'subdistrict_enc', 'type 1_enc', 
        'organization_1_enc', 'comment_len', 
        'hour', 'day_of_week', 'month'
    ]
    
    # ถ้าโมเดล มีข้อมูล feature ที่ต้องการ ให้ใช้นั้น
    if model_pkg and 'features' in model_pkg:
        required_features = model_pkg['features']
    
    # สร้าง DataFrame ที่มี Feature ที่ต้องการเท่านั้น
    X = pd.DataFrame()
    for feat in required_features:
        if feat in df_processed.columns:
            X[feat] = df_processed[feat]
        else:
            # ถ้า Feature หาไม่เจอ ให้เติม 0
            X[feat] = 0
    
    # ====== Step 5: Handle Missing Values ======
    X = X.fillna(0)
    
    # ====== Step 6: Ensure Correct Order & Data Types ======
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
    
    return X 

# โหลดโมเดลเตรียมไว้
model_package = load_model()

st.title("🚦 Traffy Fondue Analytics (Cluster & Heatmap & Prediction)")

# =========================================================
# 1. Config & Data Loading
# =========================================================

# กำหนดชื่อคอลัมน์ภาษาไทยที่คาดว่าจะเจอใน CSV (Key = ชื่อที่จะใช้ในโค้ด, Value = ชื่อหัวตารางใน CSV)
REQUIRED_COLS_CONFIG = {
    'ticket_id': 'รหัสเรื่อง (ID)',
    'comment': 'รายละเอียดปัญหา',
    'organization_1': 'หน่วยงาน (Organization)',
    'organization_2': 'หน่วยงาน (Organization)', 
    'organization_3': 'หน่วยงาน (Organization)',
    'type 1': 'ประเภทปัญหา (Type)',
    'type 2': 'ประเภทปัญหา (Type)',
    'type 3': 'ประเภทปัญหา (Type)',
    'count_reopen': 'จำนวนการเปิดซ้ำ (Reopen)',
    'star': 'คะแนน (Star)',
    'timestamp': 'วันเวลาแจ้ง (Timestamp)',
    'province': 'จังหวัด',
    'district': 'เขต/อำเภอ',
    'subdistrict': 'แขวง/ตำบล',
    'state': 'สถานะ (State)',
    'latitude': 'ละติจูด (Latitude)',
    'longitude': 'ลองจิจูด (Longitude)',
    'cluster': 'กลุ่ม (Cluster)',
    'coords': 'coords',
    'สภาพอากาศ': 'สภาพอากาศ',
    'dist_to_nearest_condo': 'dist_to_nearest_condo_km',
    'avg_price_per_sqm': 'avg_price_per_sqm'
}

@st.cache_data
def load_raw_data():
    return pd.read_csv('scrape.csv')

raw_df = load_raw_data()

@st.cache_data
def cload_cluster_df():
    try:
        return pd.read_csv("clustered_df.csv")
    except:
        return pd.DataFrame()
clusterd_df = cload_cluster_df()

if raw_df.empty:
    st.error("❌ ไม่พบไฟล์ข้อมูลหลัก (merged_data.csv หรือ clean_data2.csv)")
    st.stop()

# =========================================================
# 1.1 Auto Mapping Logic
# =========================================================

# เตรียม Dictionary สำหรับเปลี่ยนชื่อ และเก็บรายชื่อคอลัมน์ที่เจอ
rename_dict = {}
found_cols = []

# ลูปตรวจสอบคอลัมน์ตาม Config
for internal_name, csv_header in REQUIRED_COLS_CONFIG.items():
    if csv_header in raw_df.columns:
        rename_dict[csv_header] = internal_name
        found_cols.append(internal_name)
    elif internal_name in raw_df.columns:
        found_cols.append(internal_name)

# ทำการเปลี่ยนชื่อคอลัมน์
df = raw_df.rename(columns=rename_dict)

# เลือกเฉพาะคอลัมน์ที่เราต้องการใช้และมีอยู่จริงในไฟล์
df = df[found_cols]

# เช็คว่าคอลัมน์สำคัญมาครบไหม
missing_critical = []
for crit in ['latitude', 'longitude']:
    if crit not in df.columns:
        missing_critical.append(crit)

if missing_critical:
    st.warning(f"⚠️ ไม่พบคอลัมน์พิกัด: {missing_critical} ระบบอาจแสดงแผนที่ไม่ได้ (ตรวจสอบชื่อหัวตารางในไฟล์ CSV)")

# --- Process Data ---

# แปลง Timestamp
if 'timestamp' in df.columns:
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    if pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        if df['timestamp'].dt.tz is not None:
             df['timestamp'] = df['timestamp'].dt.tz_localize(None)

# แปลงตัวเลข (สำคัญมากสำหรับ Map)
for col in ['latitude', 'longitude', 'star', 'count_reopen', 'cluster', 'dist_to_nearest_condo', 'avg_price_per_sqm']:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# =========================================================
# 2. Sidebar Filters
# =========================================================
st.sidebar.header("🔍 ตัวเลือกการค้นหา (สำหรับส่วนที่ 1 & 2)")
st.sidebar.markdown("---")

if df.empty:
    st.error("ไม่เหลือข้อมูลหลังจากประมวลผล")
    st.stop()

# Filter Input
n_sample = st.sidebar.slider("1. จำนวนรายการ (Sample)", 1, 10000, min(1000, len(df)))

org_options = []
if 'organization_1' in df.columns: 
    org_options = df['organization_1'].dropna().unique()
selected_org = st.sidebar.multiselect("2. หน่วยงาน", org_options)

type_options = []
if 'type 1' in df.columns:
    type_options = df['type 1'].dropna().unique()
selected_type = st.sidebar.multiselect("3. ประเภทปัญหา", type_options)

max_reopen = int(df['count_reopen'].max()) if 'count_reopen' in df.columns and not pd.isna(df['count_reopen'].max()) else 10
reopen_range = st.sidebar.slider("4. จำนวนเปิดซ้ำ", 0, max_reopen, (0, max_reopen))

star_range = st.sidebar.slider("5. คะแนน (Star)", 0, 5, (0, 5))

# Weather filter
weather_options = []
if 'สภาพอากาศ' in df.columns:
    weather_options = df['สภาพอากาศ'].dropna().unique()
selected_weather = st.sidebar.multiselect("6. สภาพอากาศ", weather_options)

# Distance to condo filter
if 'dist_to_nearest_condo' in df.columns and df['dist_to_nearest_condo'].notna().sum() > 0:
    min_dist = float(df['dist_to_nearest_condo'].min())
    max_dist = float(df['dist_to_nearest_condo'].max())
    dist_range = st.sidebar.slider("7. ระยะห่างจากคอนโดใกล้สุด (km)", min_dist, max_dist, (min_dist, max_dist), step=0.1)
else:
    dist_range = None

# Average price per sqm filter
if 'avg_price_per_sqm' in df.columns and df['avg_price_per_sqm'].notna().sum() > 0:
    min_price = float(df['avg_price_per_sqm'].min())
    max_price = float(df['avg_price_per_sqm'].max())
    price_range = st.sidebar.slider("8. ราคาเฉลี่ยต่อตร.ม. (บาท)", min_price, max_price, (min_price, max_price), step=1000.0)
else:
    price_range = None

if 'timestamp' in df.columns and not df['timestamp'].isna().all():
    min_date = df['timestamp'].min()
    max_date = df['timestamp'].max()
else:
    min_date = datetime.now()
    max_date = datetime.now()

default_date = pd.to_datetime('2022-05-24').date()
date_range = st.sidebar.date_input("9. วันที่", [default_date, max_date])

selected_prov = st.sidebar.multiselect("10. จังหวัด", df['province'].dropna().unique() if 'province' in df.columns else [])
selected_dist = st.sidebar.multiselect("11. เขต/อำเภอ", df['district'].dropna().unique() if 'district' in df.columns else [])
selected_sub = st.sidebar.multiselect("12. แขวง/ตำบล", df['subdistrict'].dropna().unique() if 'subdistrict' in df.columns else [])
selected_state = st.sidebar.multiselect("13. สถานะ", df['state'].dropna().unique() if 'state' in df.columns else [])

# =========================================================
# Sidebar: Prediction Settings
# =========================================================
st.sidebar.markdown("---")
st.sidebar.header("🔮 ตัวเลือกแผนที่ทำนาย (สำหรับส่วนที่ 4)")

uploaded_file = st.sidebar.file_uploader("📂 อัปโหลดไฟล์ CSV เพื่อทำนาย", type=['csv'])
run_prediction = st.sidebar.button("🚀 เริ่มทำนายผล (Predict)")

n_pred_sample = st.sidebar.slider("1. จำนวนรายการทำนาย (Pred Sample)", 1, 20000, 2000, help="จำนวนจุดที่จะแสดงบนแผนที่ความเสี่ยง")
pred_dot_size = st.sidebar.slider("2. ขนาดจุดสูงสุด (Max Dot Size)", 5, 50, 15, help="ขนาดสูงสุดของวงกลมความเสี่ยง")

# --- Filtering Logic ---
filtered_df = df.copy()
# Keep original clusterd_df unfiltered for cluster section
clusterd_df_for_cluster = clusterd_df.copy()

# Filter Organization
if selected_org:
    org_conditions = False
    if 'organization_1' in filtered_df.columns:
        org_conditions = org_conditions | filtered_df['organization_1'].isin(selected_org)
    if 'organization_2' in filtered_df.columns:
        org_conditions = org_conditions | filtered_df['organization_2'].isin(selected_org)
    if 'organization_3' in filtered_df.columns:
        org_conditions = org_conditions | filtered_df['organization_3'].isin(selected_org)
    
    if isinstance(org_conditions, pd.Series):
        filtered_df = filtered_df[org_conditions]

# Filter Type (applies to 2.1, 2.2, maps only)
if selected_type:
    type_conditions = False
    if 'type 1' in filtered_df.columns:
        type_conditions = type_conditions | filtered_df['type 1'].isin(selected_type)
    if 'type 2' in filtered_df.columns:
        type_conditions = type_conditions | filtered_df['type 2'].isin(selected_type)
    if 'type 3' in filtered_df.columns:
        type_conditions = type_conditions | filtered_df['type 3'].isin(selected_type)
        
    if isinstance(type_conditions, pd.Series):
        filtered_df = filtered_df[type_conditions]

# Filter Numeric/Date
if 'count_reopen' in filtered_df.columns:
    filtered_df = filtered_df[filtered_df['count_reopen'].between(reopen_range[0], reopen_range[1])]

if 'star' in filtered_df.columns:
    filtered_df = filtered_df[filtered_df['star'].between(star_range[0], star_range[1])]

# Filter Weather (only for 2.1, 2.2 - NOT cluster)
if selected_weather and 'สภาพอากาศ' in filtered_df.columns:
    filtered_df = filtered_df[filtered_df['สภาพอากาศ'].isin(selected_weather)]

# Filter Distance to Condo (only for 2.1, 2.2 - NOT cluster)
if dist_range is not None and 'dist_to_nearest_condo' in filtered_df.columns:
    filtered_df = filtered_df[filtered_df['dist_to_nearest_condo'].between(dist_range[0], dist_range[1])]

# Filter Average Price per sqm (only for 2.1, 2.2 - NOT cluster)
if price_range is not None and 'avg_price_per_sqm' in filtered_df.columns:
    filtered_df = filtered_df[filtered_df['avg_price_per_sqm'].between(price_range[0], price_range[1])]

if 'timestamp' in filtered_df.columns and isinstance(date_range, tuple) and len(date_range) == 2:
    start_date = pd.to_datetime(date_range[0])
    end_date = pd.to_datetime(date_range[1]) + timedelta(days=1) - timedelta(seconds=1)
    filtered_df = filtered_df[(filtered_df['timestamp'] >= start_date) & (filtered_df['timestamp'] <= end_date)]

# Filter Location/State (applies to both)
if selected_prov: 
    filtered_df = filtered_df[filtered_df['province'].isin(selected_prov)]
    if 'province' in clusterd_df_for_cluster.columns:
        clusterd_df_for_cluster = clusterd_df_for_cluster[clusterd_df_for_cluster['province'].isin(selected_prov)]

if selected_dist: 
    filtered_df = filtered_df[filtered_df['district'].isin(selected_dist)]
    if 'district' in clusterd_df_for_cluster.columns:
        clusterd_df_for_cluster = clusterd_df_for_cluster[clusterd_df_for_cluster['district'].isin(selected_dist)]

if selected_sub: 
    filtered_df = filtered_df[filtered_df['subdistrict'].isin(selected_sub)]
    if 'subdistrict' in clusterd_df_for_cluster.columns:
        clusterd_df_for_cluster = clusterd_df_for_cluster[clusterd_df_for_cluster['subdistrict'].isin(selected_sub)]

if selected_state: 
    filtered_df = filtered_df[filtered_df['state'].isin(selected_state)]
    if 'state' in clusterd_df_for_cluster.columns:
        clusterd_df_for_cluster = clusterd_df_for_cluster[clusterd_df_for_cluster['state'].isin(selected_state)]

# แยก Dataframe
# analysis_df = ใช้สำหรับกราฟ 2.1 และ 2.2 (รับ filter ทั้งหมด)
analysis_df = filtered_df.copy()
# display_df = ใช้สำหรับแผนที่และตาราง (จำกัดด้วย sample)
display_df = filtered_df.head(n_sample)
# Use clusterd_df_for_cluster for cluster section (only location filters)
clusterd_df = clusterd_df_for_cluster.copy()
clusterd_df_display = clusterd_df_for_cluster.head(n_sample)

st.markdown(f"**จำนวนข้อมูลทั้งหมดที่พบ (Filter):** {len(analysis_df):,} รายการ | **แสดงผล:** {len(display_df):,} รายการ")
st.markdown("---")

# =========================================================
# 3. Visualization (Maps)
# =========================================================

st.header("1. แผนที่พิกัดและความหนาแน่น (Map Visualization)")

# กรองเอาเฉพาะที่มีพิกัดจริงเท่านั้น
if 'latitude' in display_df.columns and 'longitude' in display_df.columns:
    map_data = display_df.dropna(subset=['latitude', 'longitude'])
    map_data = map_data[(map_data['latitude'] != 0) & (map_data['longitude'] != 0)]
else:
    map_data = pd.DataFrame()

if not map_data.empty:
    mid_lat = map_data['latitude'].mean()
    mid_lon = map_data['longitude'].mean()

    view_state = pdk.ViewState(
        latitude=mid_lat,
        longitude=mid_lon,
        zoom=10,
        pitch=0,
    )

    tab_scatter, tab_heat, tab_cluster = st.tabs(["📍 Scatter Plot (แผนที่จุด)", "🔥 Heatmap (แผนที่ความร้อน)", "Cluster (แผนที่แบ่งกลุ่ม)"])

    # ---------------- TAB 1: SCATTER ----------------
    with tab_scatter:
        st.caption("แผนที่แสดงตำแหน่งรายจุด")
        
        map_data['color'] = [[255, 0, 0, 180]] * len(map_data)

        scatterplot_layer = pdk.Layer(
            "ScatterplotLayer",
            data=map_data,
            get_position='[longitude, latitude]',
            get_fill_color='color',
            get_radius=50,
            pickable=True,
            opacity=0.8,
            stroked=True,
            filled=True,
            radius_min_pixels=3,
            radius_max_pixels=10,
        )

        tooltip_fields = []
        if 'ticket_id' in map_data.columns: tooltip_fields.append("<b>ID:</b> {ticket_id}")
        if 'type 1' in map_data.columns: tooltip_fields.append("<b>Type:</b> {type 1}")
        if 'type 2' in map_data.columns: tooltip_fields.append("<b>Type:</b> {type 2}")
        if 'type 3' in map_data.columns: tooltip_fields.append("<b>Type:</b> {type 3}")
        if 'cluster' in map_data.columns: tooltip_fields.append("<b>Cluster:</b> {cluster}")
        
        tooltip_html = {
            "html": "<br/>".join(tooltip_fields) if tooltip_fields else "No Info",
            "style": {"backgroundColor": "steelblue", "color": "white"}
        }

        st.pydeck_chart(pdk.Deck(
            layers=[scatterplot_layer], 
            initial_view_state=view_state,
            tooltip=tooltip_html
        ))

    # ---------------- TAB 2: HEATMAP ----------------
    with tab_heat:
        st.caption("แผนที่แสดงความหนาแน่น (Heatmap)")
        
        heatmap_layer = pdk.Layer(
            "HeatmapLayer",
            data=map_data,
            get_position='[longitude, latitude]',
            get_fill_color='color',
            opacity=0.8,
            aggregation_name="SUM",
            radiusPixels=40,    
            intensity=1,
            threshold=0.05      
        )

        st.pydeck_chart(pdk.Deck(
            layers=[heatmap_layer], 
            initial_view_state=view_state
        ))

    # ---------------- TAB 3: CLUSTER ----------------
    with tab_cluster:
        def get_color(cluster_id):
            if cluster_id == 1: return [255, 0, 0, 200]    
            elif cluster_id == 2: return [0, 255, 0, 200]    
            elif cluster_id == 3: return [0, 0, 255, 200]    
            else: return [165, 3, 252, 200] 

        if not clusterd_df_display.empty and 'cluster' in clusterd_df_display.columns:
            clusterd_df_display['color'] = clusterd_df_display['cluster'].apply(get_color)

            view_state = pdk.ViewState(
                latitude=clusterd_df_display['latitude'].mean(),
                longitude=clusterd_df_display['longitude'].mean(),
                zoom=11,
                pitch=0
            )

            scatterplot_layer = pdk.Layer(
                "ScatterplotLayer",
                data=clusterd_df_display,
                get_position='[longitude, latitude]',
                get_fill_color='color',      
                get_radius=200,              
                radius_min_pixels=5,         
                radius_max_pixels=50,
                pickable=True,               
                opacity=0.8,
                stroked=True,
                filled=True
            )

            tooltip = {
                "html": "<b>เขต:</b> {district} <br/>"
                        "<b>Cluster:</b> {cluster} <br/>"
                        "<b>ปัญหา:</b> {first_type} <br/>"
                        "<b>สถานะ:</b> {state} <br/>"
                        "<b>Reopen:</b> {count_reopen}",
                "style": {
                    "backgroundColor": "steelblue",
                    "color": "white"
                }
            }

            st.caption("แผนที่แสดงจุดตามกลุ่ม (Cluster)")
            st.pydeck_chart(pdk.Deck(
                initial_view_state=view_state,
                layers=[scatterplot_layer],
                tooltip=tooltip
            ))
        else:
            st.info("ยังไม่มีข้อมูล Cluster ในส่วนนี้")

else:
    st.warning("⚠️ ไม่พบข้อมูลพิกัด (Latitude/Longitude) หรือข้อมูลเป็น 0 ในไฟล์หลัก")
    
st.markdown("---")

# =========================================================
# 2. Statistics & Distribution (Updated for Sliders)
# =========================================================

st.header("2. สถิติและการกระจายตัว (Distribution Analysis)")

# --- ส่วนที่ 2.1: กราฟที่ตอบสนองกับ Slider (Date, Star, Reopen) ---
st.subheader("2.1 การกระจายตัวตามตัวกรอง (Slider Filters)")
st.subheader(f"จำนวน ticket ทั้งหมด{filtered_df.shape[0]}")
st.caption("แสดงผลตามช่วงวันที่, คะแนน และจำนวนการเปิดซ้ำที่เลือก")

col_slide1, col_slide2 = st.columns(2)

# 1. กราฟ Time Series (จาก Date Range Slider)
with col_slide1:
    if 'timestamp' in analysis_df.columns:
        # Create time series data for line chart
        time_series = analysis_df.groupby(analysis_df['timestamp'].dt.date).size().reset_index(name='count')
        time_series.columns = ['timestamp', 'count']
        
        fig_time = px.line(
            time_series, 
            x='timestamp', 
            y='count',
            title="📈 ปริมาณเรื่องร้องเรียนตามช่วงเวลา (Time Distribution)",
            markers=True,
            color_discrete_sequence=['#00CC96']
        )
        fig_time.update_layout(xaxis_title="วันที่", yaxis_title="จำนวนเรื่อง", hovermode='x unified')
        st.plotly_chart(fig_time, use_container_width=True)
    else:
        st.info("ไม่พบข้อมูลวันเวลา (timestamp)")

# 2. กราฟ Star (จาก Star Slider)
with col_slide2:
    if 'star' in analysis_df.columns:
        fig_star = px.histogram(
            analysis_df, 
            x='star', 
            title="⭐ การกระจายตัวของคะแนน (Star Distribution)",
            nbins=11, 
            range_x=[-0.5, 5.5],
            color_discrete_sequence=['#FFD700']
        )
        fig_star.update_layout(xaxis_title="คะแนน (Star)", yaxis_title="จำนวนเรื่อง", bargap=0.2)
        st.plotly_chart(fig_star, use_container_width=True)
    else:
        st.info("ไม่พบข้อมูลคะแนน (star)")

# 3. กราฟ Reopen (จาก Reopen Slider)
if 'count_reopen' in analysis_df.columns:
    fig_reopen = px.histogram(
        analysis_df, 
        x='count_reopen', 
        title="🔄 จำนวนการเปิดซ้ำ (Reopen Count Distribution)",
        color_discrete_sequence=['#EF553B']
    )
    fig_reopen.update_layout(xaxis_title="จำนวนครั้งที่เปิดซ้ำ", yaxis_title="จำนวนเรื่อง")
    st.plotly_chart(fig_reopen, use_container_width=True)

# 3.1 กราฟ Top 5 Types with Most Reopen Count
if 'count_reopen' in analysis_df.columns and 'type 1' in analysis_df.columns:
    # Get top 5 types by total reopen count
    type_reopen = analysis_df.groupby('type 1')['count_reopen'].sum().nlargest(5).reset_index()
    type_reopen.columns = ['type', 'total_reopen']
    
    fig_top5_reopen = px.bar(
        type_reopen, 
        x='type', 
        y='total_reopen', 
        title="🏆 Top 5 ประเภทปัญหาที่เปิดซ้ำมากที่สุด (Top 5 Problem Types by Reopen Count)",
        color='total_reopen',
        color_continuous_scale='Reds',
        labels={'type': 'ประเภทปัญหา (Problem Type)', 'total_reopen': 'รวมจำนวนครั้งที่เปิดซ้ำ'}
    )
    fig_top5_reopen.update_xaxes(tickangle=-45)
    fig_top5_reopen.update_layout(showlegend=False)
    st.plotly_chart(fig_top5_reopen, use_container_width=True)

st.markdown("---")

# --- ส่วนที่ 2.2: กราฟประเภทปัญหา ---
st.subheader("2.2 การกระจายตัวของประเภทปัญหา (Problem Type Distribution)")

if 'type 1' in analysis_df.columns:
    if analysis_df['type 1'].notna().sum() > 0:
        problem_type_counts = analysis_df['type 1'].value_counts().reset_index()
        problem_type_counts.columns = ['type', 'count']
        fig_type = px.bar(
            problem_type_counts, 
            x='type', 
            y='count', 
            title="ประเภทปัญหา (Problem Type)",
            color='count',
            color_continuous_scale='Viridis',
            labels={'type': 'ประเภทปัญหา', 'count': 'จำนวนเรื่อง'}
        )
        fig_type.update_xaxes(tickangle=-45)
        fig_type.update_layout(showlegend=False)
        st.plotly_chart(fig_type, use_container_width=True)
    else:
        st.info("ไม่พบข้อมูลประเภทปัญหา")

st.markdown("---")

# --- ส่วนที่ 2.3: กราฟหมวดหมู่อื่นๆ ---
st.subheader("2.3 สถิติตามหมวดหมู่พื้นฐาน")

col1, col2 = st.columns(2)
other_cols = ['subdistrict', 'district', 'province', 'state', 'สภาพอากาศ'] 

for i, col_name in enumerate(other_cols):
    with (col1 if i % 2 == 0 else col2):
        if col_name in analysis_df.columns:
            if analysis_df[col_name].notna().sum() > 0:
                top_values = analysis_df[col_name].value_counts().nlargest(15).index
                filtered_chart_df = analysis_df[analysis_df[col_name].isin(top_values)]
                
                fig = px.histogram(
                    filtered_chart_df, 
                    x=col_name, 
                    title=f"การกระจายตัวของ {col_name} (Top 15)",
                    color_discrete_sequence=['#636EFA']
                )
                fig.update_xaxes(tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)

# Additional numeric distributions
st.subheader("2.4 การกระจายตัวของข้อมูลเชิงตัวเลข")
col3, col4 = st.columns(2)

with col3:
    if 'dist_to_nearest_condo' in analysis_df.columns and analysis_df['dist_to_nearest_condo'].notna().sum() > 0:
        fig_dist = px.histogram(
            analysis_df, 
            x='dist_to_nearest_condo', 
            title="📍 การกระจายระยะห่างจากคอนโดใกล้สุด (km)",
            color_discrete_sequence=['#AB63FA'],
            nbins=30
        )
        fig_dist.update_layout(xaxis_title="ระยะห่าง (km)", yaxis_title="จำนวนเรื่อง")
        st.plotly_chart(fig_dist, use_container_width=True)

with col4:
    if 'avg_price_per_sqm' in analysis_df.columns and analysis_df['avg_price_per_sqm'].notna().sum() > 0:
        fig_price = px.histogram(
            analysis_df, 
            x='avg_price_per_sqm', 
            title="💰 การกระจายราคาเฉลี่ยต่อตารางเมตร (บาท)",
            color_discrete_sequence=['#FFA15A'],
            nbins=30
        )
        fig_price.update_layout(xaxis_title="ราคาเฉลี่ย (บาท/ตร.ม.)", yaxis_title="จำนวนเรื่อง")
        st.plotly_chart(fig_price, use_container_width=True)

st.markdown("---")



# =========================================================
# 3. Cluster Analysis
# =========================================================
st.header("3. การวิเคราะห์กลุ่มปัญหา (Cluster Analysis)")

if 'cluster' in clusterd_df.columns:
    cluster_data = clusterd_df.copy()
    cluster_data['cluster'] = cluster_data['cluster'].astype(str)
    
    unique_clusters = sorted([c for c in cluster_data['cluster'].unique() if c != 'nan' and c != 'None'], key=lambda x: int(float(x)) if x.replace('.','',1).isdigit() else x)

    # 3.1 จำนวนเรื่องทั้งหมดในแต่ละ Cluster
    st.subheader("3.1 จำนวนเรื่องทั้งหมดในแต่ละ Cluster")
    total_counts = cluster_data.groupby('cluster').size().reset_index(name='count')
    total_counts = total_counts.sort_values('cluster', key=lambda col: col.map(lambda x: int(float(x)) if x.replace('.','',1).isdigit() else x))
    
    fig_total = px.bar(
        total_counts, x='cluster', y='count', color='cluster',
        title="จำนวนเรื่องทั้งหมด แบ่งตาม Cluster", labels={'cluster': 'Cluster', 'count': 'จำนวนเรื่อง'}, text_auto=True
    )
    st.plotly_chart(fig_total, use_container_width=True)

    # 3.2 สัดส่วนสถานะการดำเนินงาน
    st.subheader("3.2 สัดส่วนสถานะการดำเนินงาน (คิดเป็น % ของแต่ละ Cluster)")
    state_cluster_counts = cluster_data.groupby(['state', 'cluster']).size().reset_index(name='count')
    total_cluster_counts = state_cluster_counts.groupby('cluster')['count'].sum().reset_index(name='total_cluster_count')
    state_cluster_counts = pd.merge(state_cluster_counts, total_cluster_counts, on='cluster')
    state_cluster_counts['percentage'] = (state_cluster_counts['count'] / state_cluster_counts['total_cluster_count']) * 100
    
    fig_state_cluster = px.bar(
        state_cluster_counts, x="state", y="percentage", color="cluster",
        title="สัดส่วนสถานะการดำเนินงาน (% เทียบภายใน Cluster ตัวเอง)",
        labels={"state": "สถานะ", "percentage": "สัดส่วน (%)", "cluster": "Cluster"},
        barmode='group', text_auto='.1f'
    )
    fig_state_cluster.update_layout(yaxis_ticksuffix="%")
    st.plotly_chart(fig_state_cluster, use_container_width=True)

    # 3.3 Average Reopen
    st.subheader("3.3 ค่าเฉลี่ยการเปิดซ้ำ (Average Reopen) ราย Cluster")
    if 'count_reopen' in cluster_data.columns:
        avg_reopen = cluster_data.groupby('cluster')['count_reopen'].mean().reset_index()
        avg_reopen = avg_reopen.sort_values('cluster', key=lambda col: col.map(lambda x: int(float(x)) if x.replace('.','',1).isdigit() else x))

        fig_reopen_bar = px.bar(
            avg_reopen, x='cluster', y='count_reopen', color='cluster',
            title="ค่าเฉลี่ยจำนวนการเปิดซ้ำ (Reopen) ของแต่ละ Cluster",
            labels={'cluster': 'Cluster', 'count_reopen': 'จำนวนเปิดซ้ำเฉลี่ย'}, text_auto='.2f'
        )
        st.plotly_chart(fig_reopen_bar, use_container_width=True)

    # 3.4 Top 3
    st.subheader("3.4 รายละเอียด Top 3 ปัญหา และ เขต ของแต่ละ Cluster")
    if len(unique_clusters) > 0:
        st.markdown("##### 📌 Top 3 ประเภทปัญหา (First Type)")
        cols_type = st.columns(len(unique_clusters)) 
        for i, cluster_id in enumerate(unique_clusters):
            with cols_type[i]:
                subset = cluster_data[cluster_data['cluster'] == cluster_id]
                col_type_name = 'type 1' if 'type 1' in subset.columns else 'first_type'
                
                if col_type_name in subset.columns:
                    top_types = subset[col_type_name].value_counts().nlargest(3).reset_index()
                    top_types.columns = ['type', 'count']
                    fig_type = px.bar(top_types, x='type', y='count', title=f"Cluster {cluster_id}", text_auto=True, color_discrete_sequence=['#FF7F0E'])
                    fig_type.update_layout(xaxis_title=None, yaxis_title=None, margin=dict(l=10, r=10, t=40, b=10))
                    st.plotly_chart(fig_type, use_container_width=True)

        st.markdown("##### 🏙️ Top 3 เขต (District)")
        cols_dist = st.columns(len(unique_clusters)) 
        for i, cluster_id in enumerate(unique_clusters):
            with cols_dist[i]:
                subset = cluster_data[cluster_data['cluster'] == cluster_id]
                if 'district' in subset.columns:
                    top_dists = subset['district'].value_counts().nlargest(3).reset_index()
                    top_dists.columns = ['district', 'count']
                    fig_dist = px.bar(top_dists, x='district', y='count', title=f"Cluster {cluster_id}", text_auto=True, color_discrete_sequence=['#2CA02C'])
                    fig_dist.update_layout(xaxis_title=None, yaxis_title=None, margin=dict(l=10, r=10, t=40, b=10))
                    st.plotly_chart(fig_dist, use_container_width=True)

    # 3.5 Top 3 Problem Specific
    st.subheader("3.5 สรุปปัญหาที่พบบ่อยที่สุด (First Type) แยกตาม Cluster")
    col_problem_name = 'first_type' if 'first_type' in cluster_data.columns else 'type 1'
    if len(unique_clusters) > 0 and col_problem_name in cluster_data.columns:
        cols = st.columns(len(unique_clusters))
        for i, cluster_id in enumerate(unique_clusters):
            with cols[i]:
                subset = cluster_data[cluster_data['cluster'] == cluster_id]
                top_problems = subset[col_problem_name].value_counts().nlargest(3).reset_index()
                top_problems.columns = ['first_type', 'count']
                fig_prob = px.bar(top_problems, x='first_type', y='count', title=f"<b>Cluster {cluster_id}</b>", text_auto=True, color_discrete_sequence=['#FF5733'], height=350)
                fig_prob.update_layout(xaxis_title=None, yaxis_title=None, margin=dict(l=10, r=10, t=40, b=10), showlegend=False)
                st.plotly_chart(fig_prob, use_container_width=True)

else:
    st.info("ℹ️ ไม่พบคอลัมน์ 'cluster' ในข้อมูล จึงไม่สามารถแสดงการวิเคราะห์กลุ่มได้")

st.markdown("---")

# =========================================================
# 4. PREDICTION LOGIC
# =========================================================
st.header("4. แผนที่ความเสี่ยงการเปิดซ้ำ (Reopen Risk Prediction)")
st.caption("ทำนายความเสี่ยงจากการ Upload ไฟล์ CSV + แปลงพิกัดอัตโนมัติ")

final_pred_df = pd.DataFrame()

if uploaded_file is not None and run_prediction:
    if model_package is None:
        st.error("❌ ไม่พบไฟล์โมเดล 'traffy_model_weather.pkl' ในระบบ")
    else:
        with st.spinner("☁️ กำลังประมวลผล (แปลงพิกัด + ดึงสภาพอากาศ + ทำนายผล)..."):
            try:
                raw_up = pd.read_csv(uploaded_file)
                up_rename = {}
                for k, v in REQUIRED_COLS_CONFIG.items():
                    if v in raw_up.columns: up_rename[v] = k
                
                df_up = raw_up.rename(columns=up_rename)

                if 'coords' in df_up.columns and ('latitude' not in df_up.columns or 'longitude' not in df_up.columns):
                    try:
                        coords_split = df_up['coords'].astype(str).str.split(',', expand=True)
                        if coords_split.shape[1] >= 2:
                            df_up['longitude'] = pd.to_numeric(coords_split[0], errors='coerce')
                            df_up['latitude'] = pd.to_numeric(coords_split[1], errors='coerce')
                            st.success(f"✅ แปลงคอลัมน์ 'coords' เป็นพิกัดสำเร็จ ({len(df_up)} แถว)")
                    except Exception as e:
                        st.warning(f"⚠️ พยายามแปลง coords แล้วแต่ไม่สำเร็จ: {e}")

                if 'latitude' not in df_up.columns:
                    st.warning("⚠️ ไฟล์นี้ไม่มีคอลัมน์พิกัด (latitude/longitude หรือ coords) จะทำนายแต่ไม่แสดงแผนที่")
                    
                X_input = preprocess_for_prediction(df_up, model_package) 
                
                if model_package and 'model' in model_package:
                    model = model_package['model']
                    threshold_calc = 0.723 
                    try:
                        probs = model.predict_proba(X_input)[:, 1]
                        df_up['reopen_probability'] = probs
                        df_up['risk_level'] = df_up['reopen_probability'].apply(lambda x: 'High' if x > threshold_calc else 'Low')
                        final_pred_df = df_up
                    except Exception as e:
                        st.error(f"Prediction Error: {e} (กรุณาตรวจสอบฟังก์ชัน preprocess_for_prediction)")
                else:
                    st.error("Model format invalid")
                
            except Exception as e:
                st.error(f"เกิดข้อผิดพลาดในการอ่านไฟล์หรือทำนาย: {e}")

# Result Display
if not final_pred_df.empty:
    if 'latitude' in final_pred_df.columns and 'longitude' in final_pred_df.columns:
        pred_display = final_pred_df.head(n_pred_sample)
        map_display = pred_display.dropna(subset=['latitude', 'longitude'])
        
        if 'reopen_probability' in map_display.columns:
            map_display['adjusted_prob'] = map_display['reopen_probability'] - 0.2
            threshold_display = 0.569
            map_display['risk_color'] = map_display['adjusted_prob'].apply(
                lambda x: 'สูง (High Risk)' if x > threshold_display else 'ต่ำ (Low Risk)'
            )
            
            st.info(f"📍 ทำนายเสร็จสิ้น! แสดงผลบนแผนที่: {len(map_display):,} จุด | Threshold: {threshold_display:.4f}")
            
            if not map_display.empty:
                fig_risk = px.scatter_mapbox(
                    map_display,
                    lat="latitude", lon="longitude",
                    color="risk_color", size="reopen_probability",
                    hover_name="ticket_id" if "ticket_id" in map_display.columns else None,
                    hover_data={
                        "risk_level": True, "reopen_probability": ":.2f", "risk_color": False,
                        "district": True if 'district' in map_display.columns else False,
                        "type": True if 'type' in map_display.columns else False
                    },
                    color_discrete_map={'สูง (High Risk)': '#FF4444', 'ต่ำ (Low Risk)': '#44FF44'},
                    size_max=pred_dot_size, zoom=10, height=600,
                    title=f"แผนที่ความเสี่ยง (แดง = เสี่ยงสูง | เขียว = เสี่ยงต่ำ)"
                )
                fig_risk.update_layout(
                    mapbox_style="carto-positron",
                    mapbox_center={"lat": map_display['latitude'].mean(), "lon": map_display['longitude'].mean()},
                    margin={"r":0,"t":40,"l":0,"b":0}
                )
                st.plotly_chart(fig_risk, use_container_width=True)
        else:
            st.error("ไม่พบคอลัมน์ reopen_probability จากการทำนาย")
    else:
        st.warning("⚠️ ไม่สามารถแสดงแผนที่ได้ เนื่องจากไม่มีข้อมูล Latitude/Longitude")

    st.subheader("📋 ดาวน์โหลดผลลัพธ์")
    csv = final_pred_df.to_csv(index=False).encode('utf-8-sig')
    st.download_button(
        label="📥 ดาวน์โหลดผลการทำนาย (CSV)",
        data=csv,
        file_name="prediction_result.csv",
        mime="text/csv"
    )

else:
    st.info("👈 กรุณาอัปโหลดไฟล์ CSV ที่แถบด้านซ้าย และกดปุ่ม '🚀 เริ่มทำนายผล (Predict)'")
    st.markdown(
        """
        <div style='background-color: #f8f9fa; border: 2px dashed #ccc; border-radius: 10px; padding: 60px; text-align: center; color: #888;'>
            <h2 style='color: #ccc;'>🔮</h2>
            <h3>พื้นที่แสดงผลการทำนาย</h3>
            <p>รอข้อมูลจากการ Upload...</p>
        </div>
        """,
        unsafe_allow_html=True
    )