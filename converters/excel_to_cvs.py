import pandas as pd
import os

def convert_excel_to_csv_recursive(root_folder):
    # Tüm alt klasörlerdeki .xlsx ve .xls dosyalarını bul
    excel_files = []
    for dirpath, _, filenames in os.walk(root_folder):
        for f in filenames:
            if f.endswith('.xlsx') or f.endswith('.xls'):
                # .csv dosyası zaten var mı diye kontrol edebiliriz
                excel_files.append(os.path.join(dirpath, f))

    if not excel_files:
        print(f"'{root_folder}' dizininde Excel dosyası bulunamadı.")
        return

    print(f"Toplam {len(excel_files)} adet Excel dosyası dönüştürülüyor...")

    for excel_path in excel_files:
        # Yeni CSV dosya ismini oluştur
        base = os.path.splitext(excel_path)[0]
        csv_path = base + ".csv"

        try:
            # Excel dosyasını oku
            df = pd.read_excel(excel_path)

            # CSV olarak kaydet (indeks sütunu olmadan)
            df.to_csv(csv_path, index=False, encoding='utf-8')
            print(f"Başarılı: {os.path.basename(excel_path)} -> {os.path.basename(csv_path)}")
        except Exception as e:
            print(f"Hata oluştu ({os.path.basename(excel_path)}): {e}")

    print("\nTüm dönüştürme işlemleri tamamlandı.")

# --- KULLANIM ---
target_folder = os.path.join("../dataset", "MEMS Accelerometers")
if os.path.exists(target_folder):
    convert_excel_to_csv_recursive(target_folder)
else:
    print(f"Hata: '{target_folder}' klasörü bulunamadı.")