import pandas as pd
import os


def convert_excel_to_csv(folder_path):
    # Klasördeki tüm dosyaları listele
    files = [f for f in os.listdir(folder_path) if f.endswith('.xlsx')]

    if not files:
        print("Klasörde .xlsx dosyası bulunamadı.")
        return

    print(f"{len(files)} adet dosya dönüştürülüyor...")

    for file in files:
        # Dosya yolunu oluştur
        excel_path = os.path.join(folder_path, file)
        # Yeni CSV dosya ismini oluştur (uzantıyı değiştir)
        csv_path = os.path.join(folder_path, file.replace('.xlsx', '.csv'))

        try:
            # Excel dosyasını oku
            df = pd.read_excel(excel_path)

            # CSV olarak kaydet (indeks sütunu olmadan)
            # UTF-8 encoding Türkçe karakter sorunlarını önler
            df.to_csv(csv_path, index=False, encoding='utf-8')
            print(f"Başarılı: {file} -> {os.path.basename(csv_path)}")
        except Exception as e:
            print(f"Hata oluştu ({file}): {e}")

    print("\nTüm dönüştürme işlemleri tamamlandı.")


# --- KULLANIM ---
# Excel dosyalarının olduğu klasör yolunu buraya yazmalısın
# Örneğin: "dataset/MEMS_Accelerometers/leak"
target_folder = "dataset/MEMS Accelerometers/Leak"
convert_excel_to_csv(target_folder)