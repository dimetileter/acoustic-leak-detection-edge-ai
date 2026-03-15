from pydub import AudioSegment


def ses_birlestir(dosya_yolu, cikis_adi, tekrar_sayisi=2):
    try:
        # Ses dosyasını yükle
        ses = AudioSegment.from_wav(dosya_yolu)

        # Sesi kendi ucuna ekle (tekrar et)
        # pydub'da sesi çarpmak, onu uç uca eklemek demektir
        birlesmis_ses = ses * tekrar_sayisi

        # Sonucu dışa aktar
        birlesmis_ses.export(cikis_adi, format="wav")
        print(f"İşlem başarılı! '{cikis_adi}' dosyası oluşturuldu.")

    except Exception as e:
        print(f"Bir hata oluştu: {e}")


# --- Kullanım Örneği ---
girdi_dosyasi = "../../dataset/original_sound_datasets/leak/1.1.01.0330.wav"  # Birleştirmek istediğin dosyanın adı
cikti_dosyasi = "birlesmis_cikti.wav"

ses_birlestir(girdi_dosyasi, cikti_dosyasi, tekrar_sayisi=2)