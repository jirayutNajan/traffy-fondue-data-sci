# %%
import pandas as pd

df_downloaded = pd.read_csv("/opt/airflow/data/clean_data2.csv")

# %%
df = df_downloaded.head(100000)

# %% [markdown]
# # scape condo

# %%
import requests
import numpy as np
from scipy.spatial import cKDTree

# ---------------------------------------------------------
# PART 1: Fetch External Data (Condo/Apartment from OSM)
# ---------------------------------------------------------
print("1. Fetching Condo Data from OpenStreetMap API...")

overpass_url = "http://overpass-api.de/api/interpreter"
# Bounding Box ครอบคลุมพื้นที่ กทม. และปริมณฑล
bbox = "(13.45, 100.30, 14.05, 101.00)"

# Query: หาตึกที่เป็นที่พักอาศัยและมีชื่อระบุ
query = f"""
[out:json][timeout:60];
(
  node["residential"="condominium"]{bbox};
  way["residential"="condominium"]{bbox};
  node["building"="apartments"]{bbox};
  way["building"="apartments"]{bbox};
  node["name"~"Condo|Residence|Mansion|Apartment",i]{bbox};
);
out center;
"""

try:
    response = requests.get(overpass_url, params={'data': query})
    data = response.json()

    condo_list = []
    for item in data.get('elements', []):
        # ดึงพิกัด (รองรับทั้ง node และ way)
        lat = item.get('lat') or item.get('center', {}).get('lat')
        lon = item.get('lon') or item.get('center', {}).get('lon')

        # ดึงชื่อ (พยายามเอาภาษาไทยก่อน ถ้าไม่มีเอาอังกฤษ)
        tags = item.get('tags', {})
        name = tags.get('name:th') or tags.get('name:en') or tags.get('name')

        if lat and lon and name:
            condo_list.append({
                'condo_name': name,
                'lat': lat,
                'lon': lon
            })

    df_condo = pd.DataFrame(condo_list)
    print(f"   -> Raw data fetched: {len(df_condo)} records")

except Exception as e:
    print(f"Error fetching data: {e}")
    # ถ้า API พัง ให้สร้าง df_condo เปล่าๆ เพื่อกันโค้ด Error
    df_condo = pd.DataFrame(columns=['condo_name', 'lat', 'lon'])

# ---------------------------------------------------------
# PART 2: Clean External Data
# ---------------------------------------------------------
print("2. Cleaning Condo Data...")

if not df_condo.empty:
    # 2.1 ลบข้อมูลซ้ำ (ชื่อเดียวกันเอาไว้อันเดียว)
    df_condo.drop_duplicates(subset=['condo_name'], keep='first', inplace=True)

    # 2.2 กรอง Noise (ตัดสำนักงานขาย, โชว์รูม, ร้านค้า)
    noise_keywords = ['Sale Gallery', 'Office', 'Showroom', 'สำนักงานขาย', 'ร้าน', 'Shop', '7-Eleven', 'Market']
    pattern = '|'.join(noise_keywords)
    df_condo = df_condo[~df_condo['condo_name'].str.contains(pattern, case=False, na=False)]

    # 2.3 แปลงพิกัดเป็นตัวเลข
    df_condo['lat'] = pd.to_numeric(df_condo['lat'], errors='coerce')
    df_condo['lon'] = pd.to_numeric(df_condo['lon'], errors='coerce')

    print(f"   -> Cleaned data remaining: {len(df_condo)} records")
else:
    print("   -> No condo data available to process.")

# ---------------------------------------------------------
# PART 3: Integrate with Traffy Data & Calculate Features
# ---------------------------------------------------------
print("3. Calculating Distance and Density...")

# ใช้ตัวแปร clean_organization จากขั้นตอนก่อนหน้า
target_df = df.copy()
target_df['province'] = target_df['province'].str.strip()
target_df = target_df[target_df['province'] == 'กรุงเทพมหานคร']

# แปลงพิกัด Traffy เป็นตัวเลข
target_df['latitude'] = pd.to_numeric(target_df['latitude'], errors='coerce')
target_df['longitude'] = pd.to_numeric(target_df['longitude'], errors='coerce')
target_df = target_df.dropna(subset=['latitude', 'longitude'])

if not df_condo.empty:
    # เตรียมพิกัดสำหรับ cKDTree
    condo_coords = df_condo[['lat', 'lon']].values
    traffy_coords = target_df[['latitude', 'longitude']].values

    # สร้าง Tree ครั้งเดียว ใช้ได้ทั้งสองงาน
    tree = cKDTree(condo_coords)

    # --- TASK A: หาคอนโดที่ "ใกล้ที่สุด" (Distance) ---
    # k=1 คือเอาแค่ 1 จุดที่ใกล้ที่สุด
    dist_deg, indices = tree.query(traffy_coords, k=1)

    # แปลงองศาเป็น Km (1 องศา ≈ 111.12 km)
    target_df['dist_to_nearest_condo_km'] = dist_deg * 111.12
    target_df['nearest_condo_name'] = df_condo.iloc[indices]['condo_name'].values

    # --- TASK B: นับจำนวนคอนโดในรัศมี 1 กม. (Density) ---
    # รัศมี 1 km แปลงเป็นองศา
    radius_deg = 1 / 111.12

    # query_ball_point จะคืนค่าเป็น list ของ index ที่อยู่ในวงกลม
    indices_in_radius = tree.query_ball_point(traffy_coords, r=radius_deg)

    # นับจำนวนสมาชิกใน list
    target_df['condo_count_1km'] = [len(x) for x in indices_in_radius]

    print("   -> Calculation Completed!")
else:
    print("   -> Skipping calculation (No condo data).")

# %% [markdown]
# # scape สภาพอากาศ

# %%

def get_historical_weather(start_date, end_date, lat=13.7563, lon=100.5018):
    """
    ดึงข้อมูลจาก Open-Meteo (Archive API)
    Default Location: Bangkok (lat=13.7563, lon=100.5018)
    """
    url = "https://archive-api.open-meteo.com/v1/archive"

    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "daily": "weather_code", # ดึงรหัสสภาพอากาศรายวัน
        "timezone": "Asia/Bangkok"
    }

    response = requests.get(url, params=params)
    data = response.json()

    # สร้าง DataFrame จากข้อมูลดิบ
    df = pd.DataFrame({
        "date": pd.to_datetime(data["daily"]["time"]),
        "weather_code": data["daily"]["weather_code"]
    })

    return df

def map_weather_condition(code):
    """
    แปลง WMO Weather Code เป็น 5 กลุ่มหลักตามโจทย์
    """
    # Group 1: ฟ้าใส / แดดออก
    if code in [0]:
        return "ฟ้าใส"
    # Group 2: มีเมฆ (บางส่วน หรือ มาก)
    elif code in [1, 2, 3]:
        return "มีเมฆ"
    # Group 3: หมอก (จัดเป็นกลุ่มอื่นๆ หรือรวมกับเมฆก็ได้)
    elif code in [45, 48]:
        return "หมอก"
    # Group 4: ฝนตก (ละอองฝน, ฝนตกหนัก, ฝนเยือกแข็ง)
    elif code in [51, 53, 55, 56, 57, 61, 63, 65, 66, 67, 80, 81, 82]:
        return "ฝนตก"
    # Group 5: พายุ / หิมะ (รวมหิมะไว้เผื่อกรณีต่างประเทศ แต่ไทยหลักๆ คือพายุ)
    elif code in [95, 96, 99]:
        return "พายุฝน"
    # กรณีอื่นๆ (เช่น หิมะ code 71-77)
    else:
        return "อื่นๆ"

# --- การใช้งาน ---

# 1. กำหนดวันและดึงข้อมูล
start = "2021-09-19"
end = "2022-06-08"

# ดึงข้อมูล (ตัวอย่างพิกัด กรุงเทพฯ)
df_weather = get_historical_weather(start, end)

# 2. แปลง Code เป็นข้อความ (Map values)
df_weather['สภาพอากาศ'] = df_weather['weather_code'].apply(map_weather_condition)

# 3. จัด Format ให้เหลือแค่ 1 Column (โดยให้ Date เป็น Index เพื่อความสวยงาม)
final_df = df_weather.set_index('date')[['สภาพอากาศ']]

# แสดงผลลัพธ์
print(f"ดึงข้อมูลวันที่: {start} ถึง {end}")
print("-" * 30)
print(final_df)

# เช็คจำนวนกลุ่มสภาพอากาศที่มี
print("\nสรุปจำนวนประเภทสภาพอากาศที่พบ:")
print(final_df['สภาพอากาศ'].value_counts())

# %%
final_df['สภาพอากาศ'].head()

# %%
min_timestamp = target_df['timestamp'].min()
max_timestamp = target_df['timestamp'].max()

print(f"Minimum Timestamp: {min_timestamp}")
print(f"Maximum Timestamp: {max_timestamp}")

# %%
target_df['timestamp'] = pd.to_datetime(target_df['timestamp'], errors='coerce')
target_df['date'] = target_df['timestamp'].dt.normalize().dt.tz_localize(None)
target_df = pd.merge(target_df, final_df, left_on='date', right_index=True, how='left')

print(target_df[['timestamp', 'date', 'สภาพอากาศ']].head())

# %%
target_df.head()

# %% [markdown]
# # scape ความหนาแน่นคน

# %%
# ==============================================================================
# DATA PIPELINE (FIXED): Scrape Bangkok Population with Headers
# ==============================================================================

# 1. Scrape Data from Wikipedia (with Headers)
print("1. Scraping population data from Wikipedia...")
url = "https://en.wikipedia.org/wiki/List_of_districts_of_Bangkok"

# [FIX] เพิ่ม User-Agent เพื่อไม่ให้โดนบล็อก 403
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

try:
    # ยิง Request แบบมี Header
    response = requests.get(url, headers=headers)
    response.raise_for_status() # เช็คว่าผ่านไหม (200 OK)

    # อ่านตารางจากเนื้อหา HTML ที่ได้มา
    tables = pd.read_html(response.text)

    # ตารางแรกคือตารางเขต
    df_pop = tables[0]

    # Rename Columns
    # (ชื่อคอลัมน์ Wiki บางทีมี [1] ติดมา ต้องใช้ .str.contains หรือ rename แบบกว้างๆ)
    print("   -> Raw columns:", df_pop.columns.tolist())

    # Map ชื่อคอลัมน์ใหม่ (สังเกตตำแหน่งคอลัมน์จากหน้าเว็บ)
    # ปกติ: District (Khet) | Thai | Population | ...
    df_pop = df_pop.rename(columns={
        df_pop.columns[0]: 'district_en', # District (Khet)
        df_pop.columns[1]: 'district',    # Thai name
        df_pop.columns[2]: 'population',  # Population
        df_pop.columns[4]: 'area_sqkm'    # Area
    })

    # Clean Data
    # ลบลูกน้ำ, ลบ footnote [1], แปลงเป็นตัวเลข
    df_pop['population'] = df_pop['population'].astype(str).str.replace(r'\[.*\]', '', regex=True).str.replace(',', '')
    df_pop['area_sqkm'] = df_pop['area_sqkm'].astype(str).str.replace(r'\[.*\]', '', regex=True).str.replace(',', '')

    df_pop['population'] = pd.to_numeric(df_pop['population'], errors='coerce')
    df_pop['area_sqkm'] = pd.to_numeric(df_pop['area_sqkm'], errors='coerce')

    # สร้าง Feature: ความหนาแน่นประชากร (คน/ตร.กม.)
    df_pop['pop_density'] = df_pop['population'] / df_pop['area_sqkm']

    print(f"   -> Scraped {len(df_pop)} districts successfully.")

except Exception as e:
    print(f"❌ Error scraping: {e}")
    df_pop = pd.DataFrame()

# ---------------------------------------------------------
# 2. Merge with target_df
# ---------------------------------------------------------
if not df_pop.empty and 'target_df' in globals():
    print("\n2. Merging Population Data to Main Dataset...")

    # Clean ชื่อเขตใน df_pop ให้ตรงกับ target_df (ลบคำว่า "เขต")
    # ตัวอย่าง: "เขตพระนคร" -> "พระนคร"
    df_pop['district'] = df_pop['district'].astype(str).str.replace('เขต', '').str.strip()

    # Merge (Left Join)
    # ก่อน Merge ลบคอลัมน์เก่าออกก่อนถ้ามี (กันซ้ำ)
    if 'pop_density' in target_df.columns:
        target_df = target_df.drop(columns=['pop_density', 'population'])

    target_df = target_df.merge(df_pop[['district', 'population', 'pop_density']],
                                on='district',
                                how='left')

    # Fill Missing (สำหรับแถวที่ Join ไม่ติด)
    mean_density = target_df['pop_density'].mean()
    target_df['pop_density'] = target_df['pop_density'].fillna(mean_density)

    print("✅ Merge Completed!")
    print(target_df[['ticket_id', 'district', 'pop_density']].head())

else:
    print("❌ Cannot merge: Data not ready or 'target_df' missing.")

# %%
target_df.head()

# %% [markdown]
# # scape ราคา condo/ตรม

# %%
import json
from typing import List, Dict
import time

# Set up headers to mimic a browser request
headers = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
}

# Mapping of district codes to district names
district_mapping = {
    'TH1050': 'บางบอน',
    'TH1006': 'บางกะปิ',
    'TH1040': 'บางแค',
    'TH1005': 'บางเขน',
    'TH1031': 'บางคอแหลม',
    'TH1021': 'บางขุนเทียน',
    'TH1047': 'บางนา',
    'TH1025': 'บางพลัด',
    'TH1004': 'บางรัก',
    'TH1029': 'บางซื่อ',
    'TH1020': 'บางกอกน้อย',
    'TH1016': 'บางกอกใหญ่',
    'TH1027': 'บึงกุ่ม',
    'TH1030': 'จตุจักร',
    'TH1035': 'จอมทอง',
    'TH1026': 'ดินแดง',
    'TH1036': 'ดอนเมือง',
    'TH1002': 'ดุสิต',
    'TH1017': 'ห้วยขวาง',
    'TH1043': 'คันนายาว',
    'TH1046': 'คลองสามวา',
    'TH1018': 'คลองสาน',
    'TH1033': 'คลองเตย',
    'TH1041': 'หลักสี่',
    'TH1011': 'ลาดกระบัง',
    'TH1038': 'ลาดพร้าว',
    'TH1010': 'มีนบุรี',
    'TH1003': 'หนองจอก',
    'TH1023': 'หนองแขม',
    'TH1007': 'ปทุมวัน',
    'TH1022': 'ภาษีเจริญ',
    'TH1014': 'พญาไท',
    'TH1009': 'พระโขนง',
    'TH1001': 'พระนคร',
    'TH1008': 'ป้อมปราบศัตรูพ่าย',
    'TH1032': 'ประเวศ',
    'TH1024': 'ราษฏร์บูรณะ',
    'TH1037': 'ราชเทวี',
    'TH1042': 'สายไหม',
    'TH1013': 'สัมพันธวงศ์',
    'TH1044': 'สะพานสูง',
    'TH1028': 'สาทร',
    'TH1034': 'สวนหลวง',
    'TH1019': 'ตลิ่งชัน',
    'TH1048': 'ทวีวัฒนา',
    'TH1015': 'ธนบุรี',
    'TH1049': 'ทุ่งครุ',
    'TH1045': 'วังทองหลาง',
    'TH1039': 'วัฒนา',
    'TH1012': 'ยานนาวา',
}

def fetch_ddproperty_all_districts(max_pages_per_district: int = 2) -> pd.DataFrame:
    """
    Scrape condo data from all Bangkok districts using districtCode
    
    Parameters:
    - max_pages_per_district: Maximum number of pages to scrape per district (default: 2)
    
    Returns:
    - DataFrame with columns: [ชื่อ condo, latitude, longitude, ราคา, ตารางเมตร, ราคาต่อตารางเมตร, เขต, ชื่อเขต]
    """
    
    all_listings = []
    base_url = "https://www.ddproperty.com/_next/data/SJdltGT3KUQnE_Gt6T1yQ/%E0%B8%A3%E0%B8%A7%E0%B8%A1%E0%B8%9B%E0%B8%A3%E0%B8%B0%E0%B8%81%E0%B8%B2%E0%B8%A8%E0%B8%82%E0%B8%B2%E0%B8%A2.json"
    
    district_codes = list(district_mapping.keys())
    total_districts = len(district_codes)
    
    for district_idx, district_code in enumerate(district_codes, 1):
        district_name = district_mapping[district_code]
        print(f"\n[{district_idx}/{total_districts}] Processing: {district_code} ({district_name})")
        
        for page in range(1, max_pages_per_district + 1):
            try:
                params = {
                    'page': page,
                    'districtCode': district_code,
                    'isCommercial': False,
                    'listingType': 'sale'
                }
                
                print(f"  Fetching page {page}...", end=' ')
                response = requests.get(base_url, params=params, headers=headers, timeout=10)
                response.raise_for_status()
                
                data = response.json()
                
                # Extract listings from the correct location in the API response
                page_data = data['pageProps'].get('pageData', {})
                listings_data = page_data.get('data', {})
                listings = listings_data.get('listingsData', [])
                
                print(f"({len(listings)} listings)")
                
                for listing in listings:
                    listing_data = listing.get('listingData', {})
                    
                    # Extract price per area from the pricePerArea field
                    price_per_area_str = listing_data.get('pricePerArea', {}).get('localeStringValue', '')
                    price_per_sqm = None
                    if price_per_area_str:
                        try:
                            price_per_sqm = float(price_per_area_str.split('฿')[1].split('/')[0].replace(',', '').strip())
                        except:
                            pass
                    
                    # Extract area (ตารางเมตร) from media carousel or other fields
                    area_sqm = None
                    # Try to extract from mediaCarousel if available
                    media_carousel = listing_data.get('mediaCarousel', {})
                    if isinstance(media_carousel, dict):
                        carousel_items = media_carousel.get('items', [])
                        for item in carousel_items:
                            if isinstance(item, dict) and 'label' in item:
                                label = item.get('label', '')
                                if 'ตร.ม' in label or 'ตารางเมตร' in label:
                                    try:
                                        area_sqm = float(label.split()[0].replace(',', ''))
                                    except:
                                        pass
                    
                    # Calculate price per sqm from price and area if not provided
                    if price_per_sqm is None and listing_data.get('price', {}).get('value') and area_sqm:
                        price_per_sqm = listing_data.get('price', {}).get('value') / area_sqm
                    
                    item = {
                        'ชื่อ condo': listing_data.get('localizedTitle'),
                        'latitude': np.nan,  # Not available in this API
                        'longitude': np.nan,  # Not available in this API
                        'ราคา': listing_data.get('price', {}).get('value'),
                        'ตารางเมตร': area_sqm,
                        'ราคาต่อตารางเมตร': price_per_sqm,
                        'เขต': district_code,
                        'ชื่อเขต': district_name,
                    }
                    
                    all_listings.append(item)
                
                # Be respectful - add delay between requests
                time.sleep(1)
                
            except requests.exceptions.RequestException as e:
                print(f"  Error: {e}")
                continue
    
    # Create DataFrame with specified column order
    df = pd.DataFrame(all_listings)
    
    # Ensure all columns exist in the correct order
    columns_order = ['ชื่อ condo', 'latitude', 'longitude', 'ราคา', 'ตารางเมตร', 'ราคาต่อตารางเมตร', 'เขต', 'ชื่อเขต']
    for col in columns_order:
        if col not in df.columns:
            df[col] = np.nan
    
    df = df[columns_order]
    
    return df

# Fetch the data
print("Starting to scrape DDProperty condo data from all Bangkok districts...")
print("This may take a few minutes...")
condo_df = fetch_ddproperty_all_districts(max_pages_per_district=2)

print(f"\n{'='*60}")
print(f"SUMMARY")
print(f"{'='*60}")
print(f"Total listings scraped: {len(condo_df)}")
print(f"\nDataFrame shape: {condo_df.shape}")
print(f"\nDataFrame columns: {list(condo_df.columns)}")

print(f"\n{'='*60}")
print("First 10 rows:")
print(condo_df.head(10))

print(f"\n{'='*60}")
print("Data types:")
print(condo_df.dtypes)

print(f"\n{'='*60}")
print("Statistics for numeric columns:")
print(condo_df[['ราคา', 'ตารางเมตร', 'ราคาต่อตารางเมตร']].describe())

print(f"\n{'='*60}")
print("Listings count by district:")
print(condo_df['ชื่อเขต'].value_counts().sort_index())

# %%
# Ensure 'ราคาต่อตารางเมตร' is numeric and coerce errors to NaN
condo_df['ราคาต่อตารางเมตร'] = pd.to_numeric(condo_df['ราคาต่อตารางเมตร'], errors='coerce')

# Filter out rows where 'ราคาต่อตารางเมตร' is NaN
df_condo_price_cleaned = condo_df.dropna(subset=['ราคาต่อตารางเมตร'])

# Group by 'เขต' (code) and 'ชื่อเขต' (name) and calculate the mean of 'ราคาต่อตารางเมตร'
df_avg_condo_price_per_district = df_condo_price_cleaned.groupby(['เขต', 'ชื่อเขต'])['ราคาต่อตารางเมตร'].mean().reset_index()

# Rename the column for clarity
df_avg_condo_price_per_district.rename(columns={'ราคาต่อตารางเมตร': 'avg_price_per_sqm'}, inplace=True)

print("Average condo price per square meter by district:")

# %%
df_avg_condo_price_per_district.sort_values(by='avg_price_per_sqm', ascending=True)

# %%
# Merge df_avg_condo_price_per_district into target_df
# Ensure column names are consistent for merging
target_df = pd.merge(target_df,
                     df_avg_condo_price_per_district[['ชื่อเขต', 'avg_price_per_sqm']],
                     left_on='district',
                     right_on='ชื่อเขต',
                     how='left')

# Drop the redundant 'ชื่อเขต' column from the merge
target_df.drop(columns=['ชื่อเขต'], inplace=True)

# Fill any NaN values in 'avg_price_per_sqm' with the mean, or 0 if a mean is not appropriate
# For now, let's fill with the overall mean or 0 if mean is NaN
if 'avg_price_per_sqm' in target_df.columns:
    mean_price = target_df['avg_price_per_sqm'].mean()
    target_df['avg_price_per_sqm'].fillna(mean_price if pd.notna(mean_price) else 0, inplace=True)

print("First 5 rows of target_df with new 'avg_price_per_sqm' column:")

# %%
target_df.head()

# %%
target_df.to_csv('/opt/airflow/data/scrape.csv', index=True, encoding='utf-8')


