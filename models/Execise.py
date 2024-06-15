import pandas as pd

def load_exercise_data(file_path):
    return pd.read_csv(file_path)

def kg_to_lb(kg):
    return kg * 2.20462

def rekomendasi_aktivitas(exercise_data, target_kalori, berat_badan_kg):
    rekomendasi = {}
    
    # Konversi berat badan dari kg ke lb
    berat_badan_lb = kg_to_lb(berat_badan_kg)
    
    # Pilih kolom yang sesuai dengan berat badan
    if berat_badan_lb <= 130:
        kolom_kalori = '130 lb'
    elif berat_badan_lb <= 155:
        kolom_kalori = '155 lb'
    elif berat_badan_lb <= 180:
        kolom_kalori = '180 lb'
    else:
        kolom_kalori = '205 lb'
    
    # Hitung durasi yang dibutuhkan untuk setiap aktivitas
    for index, row in exercise_data.iterrows():
        aktivitas_name = row['Activity, Exercise or Sport (1 hour)']
        kalori_per_jam = row[kolom_kalori]
        jam_yang_dibutuhkan = target_kalori / kalori_per_jam
        rekomendasi[aktivitas_name] = "{} jam".format(round(jam_yang_dibutuhkan, 2))
    
    return rekomendasi
