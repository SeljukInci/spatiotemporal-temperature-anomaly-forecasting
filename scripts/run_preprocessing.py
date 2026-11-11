from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.preprocessing import (
    load_temperature_data,
    load_selected_cities,
    split_temperature_data,
    calculate_temperature_anomalies,
    save_processed_data,
)

CITY_KEYS = ["City", "Country", "Latitude", "Longitude"]

# 1. Ham veriyi yükle
raw_path = ROOT / "data/GlobalLandTemperaturesByCity.csv"

if not raw_path.exists():
    raise FileNotFoundError(
        f"CSV bulunamadı: {raw_path}\n"
        "raw_path değişkenini dosyanın gerçek konumuna göre düzelt."
    )

df = load_temperature_data(raw_path)

# 2. Daha önce seçilmiş 200 şehri yükle
metadata_path = ROOT / "data/processed/selected_city_metadata.csv"
selected = load_selected_cities(metadata_path)

df = df.merge(
    selected[CITY_KEYS],
    on=CITY_KEYS,
    how="inner",
    validate="many_to_one",
)

# 3. Kronolojik ayrım
train, val, test = split_temperature_data(df)

# 4. Eğitim verisinden referans hesapla
train, val, test, reference = calculate_temperature_anomalies(
    train, val, test, CITY_KEYS
)

# 5. Sonuçları doğrula
assert len(train) == 216000
assert len(val) == 26400
assert len(test) == 28800

for dataset in (train, val, test):
    assert dataset["temperature_anomaly"].notna().all()

# 6. Kaydet
save_processed_data(
    train, val, test, reference,
    output_dir=ROOT / "data/processed"
)

print("Preprocessing başarıyla tamamlandı.")