import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import pydeck as pdk

# ตั้งค่าหน้าเว็บ
st.set_page_config(page_title="Traffy Fondue Explorer (Auto Map)", layout="wide")

st.title("🚦 Traffy Fondue Analytics (Cluster & Heatmap & Prediction)")

# =========================================================
# 1. Config & Data Loading
# =========================================================

# กำหนดชื่อคอลัมน์ภาษาไทยที่คาดว่าจะเจอใน CSV (Key = ชื่อที่จะใช้ในโค้ด, Value = ชื่อหัวตารางใน CSV)
REQUIRED_COLS_CONFIG = {
    'ticket_id': 'รหัสเรื่อง (ID)',
    'comment': 'รายละเอียดปัญหา',
    'organization_1': 'หน่วยงาน (Organization)',
    'organization_2': 'หน่วยงาน (Organization)', # เผื่อกรณีไฟล์มีหลาย format
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
    'cluster': 'กลุ่ม (Cluster)'
}

@st.cache_data
def load_raw_data():
    try:
        return pd.read_csv('scrape.csv')
    except FileNotFoundError:
        try:
            return pd.read_csv('scrape.csv')
        except FileNotFoundError:
            return pd.DataFrame()

raw_df = load_raw_data()

@st.cache_data
def load_cluster_df():
    return pd.read_csv("clusterd_df.csv")
clusterd_df = load_cluster_df()

if raw_df.empty:
    st.error("❌ ไม่พบไฟล์ข้อมูลหลัก (merged_data.csv หรือ clean_data2.csv)")
    st.stop()

# =========================================================
# 1.1 Auto Mapping Logic (แทนที่ UI เดิม)
# =========================================================

# เตรียม Dictionary สำหรับเปลี่ยนชื่อ และเก็บรายชื่อคอลัมน์ที่เจอ
rename_dict = {}
found_cols = []

# ลูปตรวจสอบคอลัมน์ตาม Config
for internal_name, csv_header in REQUIRED_COLS_CONFIG.items():
    if csv_header in raw_df.columns:
        # กรณี 1: เจอชื่อภาษาไทยเป๊ะๆ ใน CSV -> สั่งเปลี่ยนชื่อ
        rename_dict[csv_header] = internal_name
        found_cols.append(internal_name)
    elif internal_name in raw_df.columns:
        # กรณี 2: CSV ถูกเปลี่ยนชื่อมาแล้ว หรือชื่อตรงกับภาษาอังกฤษอยู่แล้ว -> ใช้ได้เลย
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
for col in ['latitude', 'longitude', 'star', 'count_reopen', 'cluster']:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# =========================================================
# 2. Sidebar Filters
# =========================================================
st.sidebar.header("🔍 ตัวเลือกการค้นหา (สำหรับส่วนที่ 1 & 2)")

# --- ส่วนตั้งค่าแผนที่ ---
st.sidebar.markdown("---")
# ---------------------------------------------

if df.empty:
    st.error("ไม่เหลือข้อมูลหลังจากประมวลผล")
    st.stop()

# Filter Input
n_sample = st.sidebar.slider("1. จำนวนรายการ (Sample)", 1, 10000, min(1000, len(df)))

# ใช้ get เพื่อกัน Error กรณีคอลัมน์ไม่มี
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

if 'timestamp' in df.columns and not df['timestamp'].isna().all():
    min_date = df['timestamp'].min()
    max_date = df['timestamp'].max()
else:
    min_date = datetime.now()
    max_date = datetime.now()
    
date_range = st.sidebar.date_input("6. วันที่", [min_date, max_date])

selected_prov = st.sidebar.multiselect("7. จังหวัด", df['province'].dropna().unique() if 'province' in df.columns else [])
selected_dist = st.sidebar.multiselect("8. เขต/อำเภอ", df['district'].dropna().unique() if 'district' in df.columns else [])
selected_sub = st.sidebar.multiselect("9. แขวง/ตำบล", df['subdistrict'].dropna().unique() if 'subdistrict' in df.columns else [])
selected_state = st.sidebar.multiselect("10. สถานะ", df['state'].dropna().unique() if 'state' in df.columns else [])

# =========================================================
# NEW SIDEBAR SECTION: Prediction Settings
# =========================================================
st.sidebar.markdown("---")
st.sidebar.header("🔮 ตัวเลือกแผนที่ทำนาย (สำหรับส่วนที่ 3)")
n_pred_sample = st.sidebar.slider("1. จำนวนรายการทำนาย (Pred Sample)", 1, 20000, 2000, help="จำนวนจุดที่จะแสดงบนแผนที่ความเสี่ยง")
pred_dot_size = st.sidebar.slider("2. ขนาดจุดสูงสุด (Max Dot Size)", 5, 50, 15, help="ขนาดสูงสุดของวงกลมความเสี่ยง")

# --- Filtering Logic ---
filtered_df = df.copy()

# Filter Organization (รองรับหลายคอลัมน์ถ้ามี)
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

# Filter Type (รองรับหลายคอลัมน์ถ้ามี)
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

    if 'first_type' in clusterd_df.columns:
        clusterd_df = clusterd_df[clusterd_df['first_type'].isin(selected_type)]


if 'count_reopen' in filtered_df.columns:
    filtered_df = filtered_df[filtered_df['count_reopen'].between(reopen_range[0], reopen_range[1])]
    clusterd_df = clusterd_df[clusterd_df['count_reopen'].between(reopen_range[0], reopen_range[1])]
if 'star' in filtered_df.columns:
    filtered_df = filtered_df[filtered_df['star'].between(star_range[0], star_range[1])]

if 'timestamp' in filtered_df.columns and isinstance(date_range, tuple) and len(date_range) == 2:
    start_date = pd.to_datetime(date_range[0])
    end_date = pd.to_datetime(date_range[1]) + timedelta(days=1) - timedelta(seconds=1)
    filtered_df = filtered_df[(filtered_df['timestamp'] >= start_date) & (filtered_df['timestamp'] <= end_date)]

if selected_prov: filtered_df = filtered_df[filtered_df['province'].isin(selected_prov)]

if selected_dist: filtered_df = filtered_df[filtered_df['district'].isin(selected_dist)]
if selected_dist: clusterd_df = clusterd_df[clusterd_df['district'].isin(selected_dist)]

if selected_sub: filtered_df = filtered_df[filtered_df['subdistrict'].isin(selected_sub)]

if selected_state: filtered_df = filtered_df[filtered_df['state'].isin(selected_state)]
if selected_state: clusterd_df = clusterd_df[clusterd_df['state'].isin(selected_state)]

# แยก Dataframe
plot_df = filtered_df
display_df = filtered_df.head(n_sample)
clusterd_df_display = clusterd_df.head(n_sample)

st.markdown(f"**จำนวนข้อมูลทั้งหมดที่พบ (Filter):** {len(plot_df):,} รายการ | **แสดงผล:** {len(display_df):,} รายการ")
st.markdown("---")

# =========================================================
# 3. Visualization (Separated Maps - Original)
# =========================================================

st.header("1. แผนที่พิกัดและความหนาแน่น (Map Visualization)")

# กรองเอาเฉพาะที่มีพิกัดจริงเท่านั้น (ห้ามว่าง ห้าม NaN)
if 'latitude' in display_df.columns and 'longitude' in display_df.columns:
    map_data = display_df.dropna(subset=['latitude', 'longitude'])
    # กรองค่า 0 ออก
    map_data = map_data[(map_data['latitude'] != 0) & (map_data['longitude'] != 0)]
else:
    map_data = pd.DataFrame()

if not map_data.empty:
    mid_lat = map_data['latitude'].mean()
    mid_lon = map_data['longitude'].mean()

    # View State เริ่มต้น
    view_state = pdk.ViewState(
        latitude=mid_lat,
        longitude=mid_lon,
        zoom=10,
        pitch=0,
    )

    # สร้าง Tabs แยกกัน
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

        # สร้าง Tooltip HTML แบบ Dynamic (เช็คว่ามีคอลัมน์ไหนบ้าง)
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

    with tab_cluster:
        def get_color(cluster_id):
            if cluster_id == 1:
                return [255, 0, 0, 200]    # สีแดง (Cluster 1)
            elif cluster_id == 2:
                return [0, 255, 0, 200]    # สีเขียว (Cluster 2)
            elif cluster_id == 3:
                return [0, 0, 255, 200]    # สีน้ำเงิน (Cluster 3)
            else:
                return [165, 3, 252, 200] # สีเทา (อื่นๆ)

        # สร้างคอลัมน์สีใหม่ใน DataFrame
        clusterd_df_display['color'] = clusterd_df_display['cluster'].apply(get_color)

        # 3. กำหนดมุมมองเริ่มต้นของแผนที่
        view_state = pdk.ViewState(
            latitude=clusterd_df_display['latitude'].mean(),
            longitude=clusterd_df_display['longitude'].mean(),
            zoom=11,
            pitch=0
        )

        # 4. สร้าง Scatterplot Layer
        scatterplot_layer = pdk.Layer(
            "ScatterplotLayer",
            data=clusterd_df_display,
            get_position='[longitude, latitude]',
            get_fill_color='color',      # ใช้คอลัมน์สีที่เราสร้าง
            get_radius=200,              # รัศมีของจุด (หน่วยเป็นเมตร)
            radius_min_pixels=5,         # ขนาดต่ำสุดบนหน้าจอ
            radius_max_pixels=50,
            pickable=True,               # สำคัญมาก! ต้อง True ถึงจะแสดง Tooltip ได้
            opacity=0.8,
            stroked=True,
            filled=True
        )

        # 5. กำหนดหน้าตา Tooltip
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

        # 6. แสดงผล
        st.pydeck_chart(pdk.Deck(
            initial_view_state=view_state,
            layers=[scatterplot_layer],
            tooltip=tooltip
        ))


else:
    st.warning("⚠️ ไม่พบข้อมูลพิกัด (Latitude/Longitude) หรือข้อมูลเป็น 0 ในไฟล์หลัก")
    
st.markdown("---")

st.header("2. สถิติและการกระจายตัว")

col1, col2 = st.columns(2)
target_cols = ['subdistrict', 'district', 'province', 'state', 'star', 'count_reopen']

for i, col_name in enumerate(target_cols):
    with (col1 if i % 2 == 0 else col2):
        if col_name in plot_df.columns:
            if plot_df[col_name].notna().sum() > 0:
                fig = px.histogram(
                    plot_df, 
                    x=col_name, 
                    title=f"การกระจายตัวของ {col_name}",
                    color_discrete_sequence=['#636EFA']
                )
                st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

st.markdown("---")

# =========================================================
# NEW SECTION: Cluster Analysis (Updated with Count Chart)
# =========================================================
st.header("3. การวิเคราะห์กลุ่มปัญหา (Cluster Analysis)")

# ตรวจสอบว่ามีคอลัมน์ Cluster หรือไม่
# ใช้ plot_df ที่ผ่านการ filter มาแล้ว
if 'cluster' in clusterd_df.columns:
    # แปลง Cluster เป็น String เพื่อให้สีแยกกันชัดเจน (Discrete Color)
    cluster_data = clusterd_df.copy()
    cluster_data['cluster'] = cluster_data['cluster'].astype(str)
    
    # เรียงลำดับชื่อ Cluster (เพื่อให้กราฟเรียงสวยงาม 0, 1, 2...)
    unique_clusters = sorted([c for c in cluster_data['cluster'].unique() if c != 'nan' and c != 'None'], key=lambda x: int(float(x)) if x.replace('.','',1).isdigit() else x)

    # -------------------------------------------------------
    # 3.1 จำนวนเรื่องทั้งหมดในแต่ละ Cluster (เพิ่มใหม่ตามขอ)
    # -------------------------------------------------------
    st.subheader("3.1 จำนวนเรื่องทั้งหมดในแต่ละ Cluster")

    # นับจำนวน
    total_counts = cluster_data.groupby('cluster').size().reset_index(name='count')
    
    # เรียงลำดับตาม Cluster ID
    total_counts = total_counts.sort_values('cluster', key=lambda col: col.map(lambda x: int(float(x)) if x.replace('.','',1).isdigit() else x))
    
    fig_total = px.bar(
        total_counts, 
        x='cluster', 
        y='count', 
        color='cluster',
        title="จำนวนเรื่องทั้งหมด แบ่งตาม Cluster",
        labels={'cluster': 'Cluster', 'count': 'จำนวนเรื่อง'},
        text_auto=True
    )
    st.plotly_chart(fig_total, use_container_width=True)

    # -------------------------------------------------------
    # 3.2 Barchart ของ State (แบบ Grouped & Percentage) [Fix Logic]
    # -------------------------------------------------------
    st.subheader("3.2 สัดส่วนสถานะการดำเนินงาน (คิดเป็น % ของแต่ละ Cluster)")
    
    # 1. นับจำนวน State แยกตาม Cluster
    state_cluster_counts = cluster_data.groupby(['state', 'cluster']).size().reset_index(name='count')

    # 2. คำนวณ Total ของแต่ละ Cluster (หาผลรวม count ของ cluster นั้นๆ)
    total_cluster_counts = state_cluster_counts.groupby('cluster')['count'].sum().reset_index(name='total_cluster_count')

    # 3. Merge ข้อมูล total กลับเข้ามาในตารางหลัก
    state_cluster_counts = pd.merge(state_cluster_counts, total_cluster_counts, on='cluster')

    # 4. คำนวณ % (จำนวนใน state / จำนวนรวมใน cluster * 100)
    state_cluster_counts['percentage'] = (state_cluster_counts['count'] / state_cluster_counts['total_cluster_count']) * 100
    
    # 5. สร้างกราฟ
    fig_state_cluster = px.bar(
        state_cluster_counts, 
        x="state", 
        y="percentage",  
        color="cluster",
        title="สัดส่วนสถานะการดำเนินงาน (% เทียบภายใน Cluster ตัวเอง)",
        labels={
            "state": "สถานะ", 
            "percentage": "สัดส่วน (%)", 
            "cluster": "Cluster",
            "count": "จำนวนเรื่อง",
            "total_cluster_count": "จำนวนรวมในกลุ่ม"
        },
        barmode='group', 
        text_auto='.1f', 
        hover_data={'total_cluster_count': True, 'count': True, 'percentage': ':.2f'}
    )
    
    fig_state_cluster.update_layout(yaxis_ticksuffix="%")
    st.plotly_chart(fig_state_cluster, use_container_width=True)

    # -------------------------------------------------------
    # 3.3 Barchart ค่าเฉลี่ย Count Reopen
    # -------------------------------------------------------
    st.subheader("3.3 ค่าเฉลี่ยการเปิดซ้ำ (Average Reopen) ราย Cluster")
    
    if 'count_reopen' in cluster_data.columns:
        avg_reopen = cluster_data.groupby('cluster')['count_reopen'].mean().reset_index()
        avg_reopen = avg_reopen.sort_values('cluster', key=lambda col: col.map(lambda x: int(float(x)) if x.replace('.','',1).isdigit() else x))

        fig_reopen = px.bar(
            avg_reopen,
            x='cluster',
            y='count_reopen',
            color='cluster',
            title="ค่าเฉลี่ยจำนวนการเปิดซ้ำ (Reopen) ของแต่ละ Cluster",
            labels={'cluster': 'Cluster', 'count_reopen': 'จำนวนเปิดซ้ำเฉลี่ย'},
            text_auto='.2f'
        )
        st.plotly_chart(fig_reopen, use_container_width=True)

    # -------------------------------------------------------
    # 3.4 รายละเอียด Top 3 ของแต่ละ Cluster (Type & District)
    # -------------------------------------------------------
    st.subheader("3.4 รายละเอียด Top 3 ปัญหา และ เขต ของแต่ละ Cluster")
    
    if len(unique_clusters) > 0:
        # --- ส่วนแสดง Top 3 Type 1 ---
        st.markdown("##### 📌 Top 3 ประเภทปัญหา (First Type)")
        cols_type = st.columns(len(unique_clusters)) 
        
        for i, cluster_id in enumerate(unique_clusters):
            with cols_type[i]:
                subset = cluster_data[cluster_data['cluster'] == cluster_id]
                # เช็คทั้ง type 1 และ first_type เพื่อความชัวร์
                col_type_name = 'type 1' if 'type 1' in subset.columns else 'first_type'
                
                if col_type_name in subset.columns:
                    top_types = subset[col_type_name].value_counts().nlargest(3).reset_index()
                    top_types.columns = ['type', 'count']
                    
                    fig_type = px.bar(
                        top_types, x='type', y='count',
                        title=f"Cluster {cluster_id}", text_auto=True,
                        color_discrete_sequence=['#FF7F0E']
                    )
                    fig_type.update_layout(xaxis_title=None, yaxis_title=None, margin=dict(l=10, r=10, t=40, b=10))
                    st.plotly_chart(fig_type, use_container_width=True)

        # --- ส่วนแสดง Top 3 District ---
        st.markdown("##### 🏙️ Top 3 เขต (District)")
        cols_dist = st.columns(len(unique_clusters)) 
        
        for i, cluster_id in enumerate(unique_clusters):
            with cols_dist[i]:
                subset = cluster_data[cluster_data['cluster'] == cluster_id]
                if 'district' in subset.columns:
                    top_dists = subset['district'].value_counts().nlargest(3).reset_index()
                    top_dists.columns = ['district', 'count']
                    
                    fig_dist = px.bar(
                        top_dists, x='district', y='count',
                        title=f"Cluster {cluster_id}", text_auto=True,
                        color_discrete_sequence=['#2CA02C']
                    )
                    fig_dist.update_layout(xaxis_title=None, yaxis_title=None, margin=dict(l=10, r=10, t=40, b=10))
                    st.plotly_chart(fig_dist, use_container_width=True)

    # -------------------------------------------------------
    # 3.5 Top 3 ปัญหา (First Type) แยกกราฟเฉพาะ (ตามที่ขอเพิ่ม)
    # -------------------------------------------------------
    st.subheader("3.5 สรุปปัญหาที่พบบ่อยที่สุด (First Type) แยกตาม Cluster")
    
    # ใช้ logic เดียวกับด้านบนแต่แยกส่วนออกมา
    col_problem_name = 'type 1' # Default
    if 'first_type' in cluster_data.columns:
        col_problem_name = 'first_type'
    elif 'type 1' in cluster_data.columns:
        col_problem_name = 'type 1'
        
    if len(unique_clusters) > 0 and col_problem_name in cluster_data.columns:
        cols = st.columns(len(unique_clusters))
        
        for i, cluster_id in enumerate(unique_clusters):
            with cols[i]:
                subset = cluster_data[cluster_data['cluster'] == cluster_id]
                
                top_problems = subset[col_problem_name].value_counts().nlargest(3).reset_index()
                top_problems.columns = ['first_type', 'count']
                
                fig_prob = px.bar(
                    top_problems,
                    x='first_type',
                    y='count',
                    title=f"<b>Cluster {cluster_id}</b>",
                    text_auto=True,
                    color_discrete_sequence=['#FF5733'],
                    height=350
                )
                
                fig_prob.update_layout(
                    xaxis_title=None, 
                    yaxis_title=None, 
                    margin=dict(l=10, r=10, t=40, b=10),
                    showlegend=False
                )
                st.plotly_chart(fig_prob, use_container_width=True)

else:
    st.info("ℹ️ ไม่พบคอลัมน์ 'cluster' ในข้อมูล จึงไม่สามารถแสดงการวิเคราะห์กลุ่มได้")

# =========================================================
# 5. NEW SECTION: Reopen Risk Visualization
# =========================================================
st.header("4. แผนที่ความเสี่ยงการเปิดซ้ำ (Reopen Risk Prediction)")
st.caption("แสดงผลจากไฟล์ prediction_result.csv (สีแดง = ความเสี่ยงสูง, สีน้ำเงิน = ความเสี่ยงต่ำ)")

# 5.1 Load Prediction Data
@st.cache_data
def load_prediction_data():
    try:
        # อ่านไฟล์ prediction_result.csv
        df_pred = pd.read_csv('prediction_results.csv')
        
        # ตรวจสอบคอลัมน์ที่จำเป็น
        required_pred_cols = ['latitude', 'longitude', 'reopen_probability', 'risk_level', 'ticket_id', 'type', 'district']
        missing_cols = [col for col in required_pred_cols if col not in df_pred.columns]
        
        if missing_cols:
             st.error(f"❌ ไฟล์ prediction_result.csv ขาดคอลัมน์ที่จำเป็น: {missing_cols}")
             return pd.DataFrame()
             
        return df_pred
    except FileNotFoundError:
        st.warning("⚠️ ไม่พบไฟล์ 'prediction_result.csv' กรุณาอัปโหลดไฟล์เพื่อดูการวิเคราะห์ความเสี่ยง")
        return pd.DataFrame()

pred_df = load_prediction_data()

# 5.2 Process & Plot Prediction Data
if not pred_df.empty:
    # แปลงเป็นตัวเลขและกรอง NaN
    pred_df['latitude'] = pd.to_numeric(pred_df['latitude'], errors='coerce')
    pred_df['longitude'] = pd.to_numeric(pred_df['longitude'], errors='coerce')
    pred_df = pred_df.dropna(subset=['latitude', 'longitude', 'reopen_probability'])
    
    # --- APPLY SLIDERS HERE ---
    # ใช้ Slider จาก Sidebar มาตัดจำนวนข้อมูล
    pred_df = pred_df.head(n_pred_sample)
    
    if pred_df.empty:
         st.warning("พบไฟล์ Prediction แต่ไม่มีข้อมูลพิกัด (Latitude/Longitude) ที่ถูกต้อง")
    else:
        # แสดงจำนวนข้อมูลที่กำลังพล็อต
        st.info(f"📍 กำลังแสดงผล: {len(pred_df):,} รายการ (จากทั้งหมดในไฟล์) | ขนาดจุดสูงสุด: {pred_dot_size}px")

        # สร้าง Scatter Mapbox ด้วย Plotly Express
        # ใช้ 'reopen_probability' เป็นตัวกำหนดสี (Continuous Color Scale)
        # สีแดง (Red) = โอกาสสูง, สีน้ำเงิน (Blue) = โอกาสต่ำ (ใช้ RdBu_r reversed)
        fig_risk = px.scatter_mapbox(
            pred_df,
            lat="latitude",
            lon="longitude",
            color="reopen_probability", # สีตามความน่าจะเป็น
            size="reopen_probability",  # ขนาดตามความน่าจะเป็น (ยิ่งเสี่ยงยิ่งใหญ่)
            hover_name="ticket_id",
            hover_data={
                "latitude": False,
                "longitude": False,
                "type": True,
                "risk_level": True,
                "district": True,
                "reopen_probability": ":.2f" # แสดงทศนิยม 2 ตำแหน่ง
            },
            color_continuous_scale=px.colors.sequential.RdBu_r, # โทนสี แดง-น้ำเงิน
            size_max=pred_dot_size, # <--- ใช้ค่าจาก Slider ขนาดจุดตรงนี้
            zoom=10,
            height=600,
            title="แผนที่แสดงความเสี่ยงการถูกเปิดซ้ำ (Reopen Risk Map)"
        )

        # ตั้งค่าจุดกึ่งกลางแผนที่
        mid_lat_pred = pred_df['latitude'].mean()
        mid_lon_pred = pred_df['longitude'].mean()
        fig_risk.update_layout(
            mapbox_style="carto-positron", # ใช้แผนที่พื้นหลังแบบสว่าง
            mapbox_center={"lat": mid_lat_pred, "lon": mid_lon_pred},
            margin={"r":0,"t":40,"l":0,"b":0}
        )

        st.plotly_chart(fig_risk, use_container_width=True)

        # *** ส่วนแสดงตาราง Prediction ถูกลบออกแล้วตามคำขอ ***

st.markdown("---")