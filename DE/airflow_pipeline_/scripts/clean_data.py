# %%
import pandas as pd

df = pd.read_csv('/opt/airflow/data/bangkok_traffy.csv')

# %%
print(df.shape)

# %%

# %%
for col in df.columns:
  print(df[df[col].isna()].shape, col)

# %%
drop_na = df.dropna(subset=['type', 'organization', 'subdistrict', 'district', 'province', ])

# %%
drop_na.shape

# %% [markdown]
# # Clear แต่ละ column

# %% [markdown]
# ## Column province

# %%
print(drop_na['province'].unique())
print(drop_na['province'].unique().shape)

# %%
# ลบ คำว่า จังหวัด นำหน้า
drop_na['province'] = (drop_na['province'].str.replace('จังหวัด', '', regex=False))
print(drop_na['province'].unique())
print(drop_na['province'].unique().shape)

# %%
province_mapping = {
    # รวมเข้ากับ 'กรุงเทพมหานคร'
    'กรุงเทพฯ': 'กรุงเทพมหานคร',
    'Bangkok': 'กรุงเทพมหานคร',

    # จัดการกับคำย่อ 'จ.'
    'จ.ฉะเชิงเทรา': 'ฉะเชิงเทรา',
    'จ.สุพรรณบุรี': 'สุพรรณบุรี',

    # จัดการกับค่าว่าง
    '': 'ไม่ระบุ' # เปลี่ยนค่าว่างเป็น 'ไม่ระบุ'
}

drop_na['province'] = drop_na['province'].replace(province_mapping)

drop_na['province'].unique()

# %%
clean_province = drop_na

# %% [markdown]
# ## เช็คความถูกต้องของ column

# %%
for col in clean_province:
  print(col, clean_province[clean_province[col].isna()].shape)

# %% [markdown]
# ⬆️⬆️⬆️⬆️ ลบ na ของ column ที่จำเป็นออกหมดแล้ว

# %% [markdown]
# ## แยก coords เป็น Lat Long

# %%
lat_long = clean_province

# %%
split_coords = lat_long['coords'].str.split(',', expand=True)

#latitude and longitude
lat_long['latitude'] = split_coords[1]
lat_long['longitude'] = split_coords[0]

# %%

# %%
for col in lat_long.columns:
  print(col, lat_long[lat_long[col].isna()].shape)

# %%
for col in lat_long.columns:
  print(col, lat_long[col].unique())
  print(lat_long[col].unique().shape)

# %% [markdown]
# # จัดการ type

# %%
# 1. สร้าง column ใหม่เพื่อเก็บ "จำนวน type" ในแต่ละแถว
# หลักการ: ลบปีกกา {} ออก -> แยกคำด้วยคอมม่า -> นับจำนวนสมาชิกที่แยกได้
lat_long['type_count'] = lat_long['type'].str.strip('{}').str.split(',').apply(len)

# --- คำตอบข้อที่ 1 และ 2: Row ที่มี type มากที่สุด ---
# หา index ของแถวที่มีค่า type_count สูงที่สุด
max_idx = lat_long['type_count'].idxmax()
row_max = lat_long.loc[max_idx]

print(f"Row ที่มี type มากที่สุดคือ Index: {max_idx}")
print(f"จำนวน type ใน Row นี้คือ: {row_max['type_count']}")
print(f"ค่าใน Row นี้คือ: {row_max['type']}")

print("-" * 30)

# --- คำตอบข้อที่ 3: ค่าเฉลี่ยของ type ต่อ 1 row ---
average_type = lat_long['type_count'].mean()

print(f"ค่าเฉลี่ยของ type ต่อ 1 row คือ: {average_type:.2f} ประเภท")

# %%
# สมมติว่า df คือ DataFrame ของคุณ
# ขั้นตอนที่ 1: ลบปีกกา {} และแยกข้อมูลด้วยคอมม่า (,)
# expand=True จะทำให้ผลลัพธ์ออกมาเป็น DataFrame แยก column ให้เลย
split_types = lat_long['type'].str.strip('{}').str.split(',', expand=True)

# ขั้นตอนที่ 2: บังคับให้เอาแค่ 3 column แรก (0, 1, 2)
# ถ้าข้อมูลเดิมมีแค่ 1 หรือ 2 ตัว columns ที่ขาดจะถูกเติมด้วย NaN (null)
# ถ้าข้อมูลเดิมมีเกิน 3 ตัว ส่วนเกินจะถูกตัดทิ้ง
split_types = split_types.reindex(columns=[0, 1, 2])

# ขั้นตอนที่ 3: ตั้งชื่อ column ใหม่ และนำไปใส่ใน DataFrame หลัก
lat_long[['type 1', 'type 2', 'type 3']] = split_types

# %%
clean_type = lat_long


# %%
clean_type[clean_type['type 1'] == ""].shape

# %%
clean_type = clean_type[clean_type['type 1'] != ""]
clean_type.shape

# %%
clean_type.head()

# %%
clean_type = clean_type.drop(columns=['type', 'type_count'])

# %%

# %%
clean_type.shape

# %%
clean_organization = clean_type

# %%
# สมมติว่า DataFrame ของคุณชื่อ df
# 1. สร้างคอลัมน์ใหม่เพื่อนับจำนวน organization ในแต่ละแถว
# โดยการสั่ง split ด้วย ',' และนับความยาวของลิสต์ที่ได้
clean_organization['org_count'] = clean_organization['organization'].str.split(',').apply(len)

# 2. แสดง row ที่มี organization มากสุด
row_with_max_org = clean_organization[clean_organization['org_count'] == clean_organization['org_count'].max()]
print("Row ที่มี organization มากที่สุด:")

# 3. ค่าเฉลี่ย organization ต่อ 1 row
average_org = clean_organization['org_count'].mean()
print(f"ค่าเฉลี่ย organization ต่อ 1 row: {average_org}")

# %%
# 1. แยกข้อความด้วย comma (,) และกระจายออกเป็น DataFrame ชั่วคราว
# expand=True จะเปลี่ยน list ให้เป็น columns
split_df = clean_organization['organization'].str.split(',', expand=True)

# 2. สร้าง Column organization_1, 2, 3
# เราต้องเช็คด้วยว่า split_df มีคอลัมน์ครบมั้ย (ป้องกัน Error กรณีข้อมูลทุกแถวมีแค่ 1 organization)

clean_organization['organization_1'] = split_df[0] if 0 in split_df.columns else None
clean_organization['organization_2'] = split_df[1] if 1 in split_df.columns else None
clean_organization['organization_3'] = split_df[2] if 2 in split_df.columns else None

# %%
clean_organization.columns
#org_count organization

# %%
clean_organization = clean_organization.drop(columns=['org_count', 'organization'])

# %%
clean_organization = clean_organization.drop(columns=['coords'])

# %%

# %%
for col in clean_organization.columns:
  print(col, clean_organization[clean_organization[col].isna()].shape)

# %%
clean_organization.to_csv('/opt/airflow/data/clean_data2.csv', index=True, encoding='utf-8')


