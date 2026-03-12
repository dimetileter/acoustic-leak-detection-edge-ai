### Uç Uca Ses Ekleyerek Siyah Alandan Kurtulma Denemesi

Spektogram reismlerinde meydana gelen "siyah boş alan" problemi için resmin bu alanını doldurmaya odaklanıldı.
Bunun için aynı ses dosyası uç uca birleştirildi (birlesmis_cikti.wav) ve oluşan yeni ses dosyası görüntüye çevirildi.
Uç uca eklenen sesin görseli (birlesmis_cikti.png) ile orijinal halinin görseli (1.1.01.0330.png) hemen hemen aynı sonucu verdi.
_**Siyah alanda değişme meydana gelmedi.**_

### Librosa Freakans Küçültme Denemesi
Ses frekansı ilk başta 8000 olarak ayarlı idi. Frekasnı 4000'e düşürünce kaotik biçimde siyah alanın daraldığı gözlemlendi.
Daha sonrasında kademeli olarak frekans düşürmesi yapılarak optimum frekans bulundu. Librosa frekans ayarlaması şimdilik dataset probleminin
çözümü olabilir mi?
**Siyah alan problemi çözüldü**
