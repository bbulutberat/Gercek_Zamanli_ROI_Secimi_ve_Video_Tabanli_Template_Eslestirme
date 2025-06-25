# Gerçek Zamanlı ROI Seçimi ve Video Tabanlı Template Eşleştirme

Bu proje, kullanıcıdan bir video akışı üzerinde belirli bir alan (ROI) seçmesini ister ve bu seçilen alanı (şablon) tüm video boyunca arar. Eşleşme bulunduğunda, sistem videoyu duraklatır ve eşleşen bölgeyi görsel olarak işaretler.

## Projenin Amacı

- Video üzerinden bir karede manuel olarak ROI (Region of Interest) seçmek
- Seçilen bölgeyi şablon (template) olarak alıp videonun tamamında bu şablonun benzerini şablon eşleştirme (template matching)yöntemiyle aramak
- Eşleşme tespit edildiğinde videoyu durdurmak ve eşleşen alanı dikdörtgenle göstermek

## Nasıl Çalışır?

- Video açılır ve kullanıcı `m` tuşuna basana kadar video oynatılır.
- Kullanıcı `m` tuşuna bastığında, durdurulan kare üzerinde fareyle bir bölge seçilir (ROI).
- Seçilen bölge `template` olarak kaydedilir.
- Video baştan oynatılır.
- Her karede `cv2.matchTemplate()` fonksiyonu ile eşleşme kontrolü yapılır.
- Eşleşme değeri belirlenen eşiğin altına düştüğünde, eşleşme başarılı sayılır ve:
   - Eşleşme alanı yeşil dikdörtgenle çizilir.
   - Video durdurulur (`cv2.waitKey(0)`).
   - Kullanıcı `q` tuşuna basarak çıkış yapabilir.

## Template Matching
- Template Matching daha büyük bir görüntüde bir template 
görüntüsünün konumunu aramak ve bulmak için kullanılan bir yöntemdir. 
OpenCV, bu amaç için cv2.matchTemplate() fonksiyonunu kullanır. Template 
görüntüsünü giriş görüntüsünün üzerine kaydırır ve template görüntüsünün 
altındaki girdi görüntüsünü template görüntüsü ile karşılaştırır.

**Parametreler**
- Cv2.matchTemplate() fonksiyonu sırasıyla 3 parametre alır;
    - İmage = Nesnenin bulunduğu büyük resim
    - Template = Algılamak istediğimiz nesne
    - Template Matching Yöntemleri:
        | Yöntem             | Skor           | Ne Zaman Kullanılır?                                  |
        | ------------------ | -------------- | ----------------------------------------------------- |
        | `TM_SQDIFF`        | Düşük = iyi    | Birebir benzer şablonlar, düşük gürültü               |
        | `TM_SQDIFF_NORMED` | Düşük = iyi    | Birebir eşleşmeler ama ışık farkına daha dayanıklı    |
        | `TM_CCORR`         | Yüksek = iyi   | Arka plan sabitse, kontrast iyi ise                   |
        | `TM_CCORR_NORMED`  | Yüksek = iyi   | Aynı ama parlaklık değişimlerine karşı daha dayanıklı |
        | `TM_CCOEFF`        | Yüksek = iyi   | Şekil aynı, ışık farklıysa (gölge/parlaklık)          |
        | `TM_CCOEFF_NORMED` | Yüksek = iyi   | En kararlı yöntem, çoğu durumda işe yarar             |

**minMaxLoc()**
- `cv2.minMaxLoc(res)` fonksiyonu, `cv2.matchTemplate()` fonksiyonunun çıktısı eşleşme bilgileri arasından en küçük ve en büyük skorları ve bunların konumlarını döndürür.  

----------------------------------------------------------------------------------------------------------------------------------------
## KOD AÇIKLAMASI 

```
import cv2
import numpy as np

class ROI():
    
    def __init__(self):
        #video dosyasını açar.
        self.cap = cv2.VideoCapture("trafik.mp4")

    def video(self):

        #Video oynatılması için döngü.
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break #video bittiğinde döngüden çık

            #video karesini oynat.
            cv2.imshow("video",frame)
            key = cv2.waitKey(30)

            # Kullanıcı 'm' tuşuna bastığında ROI seçimi yapılır.
            if key == ord("m"):

                # Mouse ile ROI seçimi yapılır.
                roi = cv2.selectROI("Roi", frame)
                # Bu kare daha sonra Template Matching yapmak için saklanır.
                self.img = frame
                cv2.destroyAllWindows()
                break
            
            # Kullanıcı 'q' tuşuna basarsa çıkılır.
            if key ==ord("q"):
                cv2.destroyAllWindows()
                break
        # Videoyu serbest bırak.
        self.cap.release()
        self.template(roi)
        
    
    def template(self, roi):

        # ROI yapılan framden template çıkarılır.
        x, y, w, h = roi
        template = self.img[y:y+h, x:x+w]
        self.templateMatch(template)

    def templateMatch(self, template):

        # Videoyu baştan oynatmak için tekrar video açılır.
        self.cap = cv2.VideoCapture("trafik.mp4")

        #template görüntüsü griye çevirilir.
        template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        
        # Nesne bulunduğunda dikdörtgen içine almak için nesnenin yükseklik ve genişlik bilgisi alınır
        w,h = template_gray.shape[::-1]

        while True:
            ret, frame = self.cap.read()

            # Her frame griye çevirilir.
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Bulunulan framde template matching işlemleri yapılır.
            res = cv2.matchTemplate(frame_gray, template_gray, cv2.TM_SQDIFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

            # Videuyo göster.
            cv2.imshow("video", frame)
            cv2.waitKey(30)

            # Uygun eşleşme bulunduysa videoyu duraklat ve dikdörtgeni çiz.
            if min_val < 0.005:
                bottom_right = (min_loc[0] + w, min_loc[1] + h)
                cv2.rectangle(frame, min_loc, bottom_right, (0,255,0), 2)
                cv2.imshow("video", frame)
                key = cv2.waitKey(0)

                if key == ord("q"):
                    break
            
            if not ret:
                break
        
        self.cap.release()

        
if __name__ == "__main__":
    baslat = ROI()
    baslat.video()
``` 



