import os

"""
    Programın genel çalışması ile bir alakası yok. 
    Görsele çevirilen dosyaların eksiksiz sayıda olup olmadığını teyit etmek için oluşturdum.
"""

leak_dir = "spectrogram_pool/leak"
no_leak_dir = "spectrogram_pool/no_leak"

counter = 0
for leak_dir in os.listdir(leak_dir):
    counter += 1

print(f"leak: {counter}")

counter = 0
for no_leak_dir in os.listdir(no_leak_dir):
    counter += 1

print(f"leak: {counter}")
print("[Tamamlandı]: Sonuç 70-70 dağılmışsa işlem doğrudur")