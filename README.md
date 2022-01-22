# Average faces with OpenCV (Python)

Calculate an average face from multiple images using the machine learning library **dlib** and the computer vision toolkit **OpenCV**. For this example, we'll use images of Bavarian politicians. These instructions and the some of the code build on Satya Mallick's excellent introduction to computer vision: [Learn OpenCV](http://www.learnopencv.com/average-face-opencv-c-python-tutorial/).

Bu çalışmada, bir grup resim verildiğinde; resimlerin okunması, insan yüzü içeriyorsa tespit edilmesi, tespit edilen yüzlerin özelliklerinin çıkarılması ve ortalamaları alınarak tek bir yüzün grubu temsil edecek şekilde bir araya getirilmesi aşamaları, makine öğrenme kütüphanesi **Dlib** ve görüntü işleme kütüphanesi **openCV** kullanılarak gerçekleştirilecektir.
Kodların yazılması ve genel algoritma akışı için Satya Mallick'in [Learn OpenCV](http://www.learnopencv.com/average-face-opencv-c-python-tutorial/) blog yazılarından faydalanılmıştır.

kodlarda düzenleme ve iyileştirme yapılmıştır.


## Data
I put together a list of all members of the Bavarian parliament, which I scraped from their [website](https://www.bayern.landtag.de/politicians/politicians-von-a-z/). The dataset contains unique IDs which we'll use to download the image of each politician.

çalışmada kullanmak için resimler, Steffen Kühne'nin github hesabından örnek alınmıştır. [https://github.com/stekhn]

## Requirements

[Python 2](https://www.python.org/downloads/) is required for running the scripts,
[OpenCV Python](https://pypi.python.org/pypi/opencv-python) uses pre-compiled OpenCV binary and can be installed using the Python package manager 
[pip](https://pypi.python.org/pypi/pip). Make sure to remove previous or other versions of OpenCV, to avoid conflicts. 

[Dlib](http://dlib.net/), which we'll use for landmark extraction, requires CMake to build:

scriptleri çalıştırmak için python2 kullanıldı. Görüntüleri okumak, göstermek ve Landmarkları çıkarmak için ise opencv ve dlib kütüphaneleri kullanıldı. 

Dlib kütüphanesini kurmak ve kullanmak için çalışma ortamımızda cmake kurulu olmalıdır. Kurulu değilse kurulum yapmak için pip kullanılabilir.

```
$ pip install cmake
```
To avoid Python dependency trouble, we'll use the Python virtual environment wrapper [virtualenv](https://virtualenv.pypa.io/en/stable/).
python ve diğer kütüphanelerin versiyon farklarından dolayı çakışma olmaması için sanal çalışma ortamı kurulması tavsiye edilir

## Setup

Update your Python package manager:
Sanal bir çalışma ortamı kurmak için python paketleri güncellenir daha sonra virtualenv kurulu değilse pip ile kurulum yapılabilir
```
pip install --upgrade pip
```
```
pip install virtualenv
```

Create a new virtual environment:
sanal bir çalışma ortamı kuruyoruz, isim için venv yerine proje ismi yazılabilir

```
$ virtualenv venv
```

Activate the virtual environment:
çalışma ortamını aktif etmek için

```
$ source venv/bin/activate
```
veya cd komutuyla çalışma ortamına girdikten sonra kısaca
```
$ source ./bin/activate
```
kullanılabilir..

Check if the Python virtual environment is set up correctly:
kurulumun doğruluğu kontrol edilir.

```
$ which python
/Users/your-username/Development/venv/env/bin/python
```

daha sonra çalışma ortamına, scriptlerin çalışması için gerekli olan eklentiler(kütüphene ve paketler) kurulur.
kurulumu yapmak için requriements.txt dosyasını çalışma ortamına indirdikten sonra pip ile çalıştırılabilir.
Install dependencies: 

```
$ pip install -r requirements.txt
```

## Download images

ortalaması istenilen resimler toplanılır ve çalışma ortamına eklenir. Eğer hazır resimlerle devem edilecekse [https://github.com/stekhn/average-faces-opencv] adresinden download.py dosyası indirilerek çalıştırılabilir

```
"https://www.bayern.landtag.de/images/politicians/" + "555500000394" + ".jpg"
```

## Extract face landmarks

The script tries to find human faces in an image and extract 68 landmarks. These are points on the face such as the corners of the mouth, along the eyebrows, on the eyes, and so forth. We'll need this landmarks to map the different faces onto each other.

The script needs a pre-trained model for predicting these features, which is available for download (~ 60 MB):

Toplanılan resimlerdeki yüzlerin ortalamasını almak için öncelikle yüzlerin tespit edilmesi daha sonra da her bir yüz için "Facial landmarks" denilen yüz ile ilgili kimliksel noktaların tespit edilerek çıkarılması gerekir.

landmarkları çıkarmak için makina öğrenmesinden faydalanılır. Kendi modelini eğitmek istemeyenler için Dlib resmi sayfasında önceden eğitilmiş face_landmark_predictor bulunmaktadır. indirilip kullanılabilir. 

projeye dahil etmek için wget ile indirilebilir

```
$ wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
```

daha sonra arşiv dosyası çıkarılır
Unzip the shape predictor (~ 95 MB):

```
$ bzip2 -dk shape_predictor_68_face_landmarks.dat.bz2
```
extract.py dosyasına arguman olarak verilir ve python2 yardımıyla scriptler çalıştırılır. 
```
$ python extract.py shape_predictor_68_face_landmarks.dat ./images
```
burda önemli olan nokta şu tüm yüzlerin land_markları çıkarılarak kendi isimleri ile aynı dosyaya kaydedilir ve tekrar tekrar kullanılabilir. Tabi dosyaya ekleme veya dosyadan silme işlemleri yapılmadığı sürece.

The extracted landmarks will be saved as list of xy coordinates in the same folder as the images, using a ".txt" extension.



## Average faces

Bring the images to the same size and roughly align the images using the position of the eyes. Other features of the face might be misaligned. Therefore, we'll use a bounding box to triangulate the landmark points ([Delaunay Triangulation](http://www.learnopencv.com/delaunay-triangulation-and-voronoi-diagram-using-opencv-c-python/)). These triangles can then be warped to match the other triangles, so the faces line up neatly. Finally the images will be blended together by applying some transparency.

To run the script, provide the path to your image folder. The folder should contain both images (.jpg) and landmarks (.txt). Optionally, you can specify the desired output size for the output image (width, height):

Resimleri düzgün bir şekilde eşlenmesi için öncelikle resim boyutları eşitlenir. Sonra landmarklar kullanılarak yüz resimleri üçgensel parçalara ayrılır. Yüzler göz_köşeleri eşleşecek şekilde üçgensel bölgeler eşleştirilir. Alfa_blending özelliği kullanılarak yüzlerin saydamlığı ayarlanılır ve tüm yüzlerin landmarklarının ortalaması alınarak oluşturulmuş olan taslak üzerinde birleştirme yapılır.

average.py dosyası ortalaması alınacak ve daha önceden landmarkları belirlenmiş olan ile beraber çalıştırılır.
"çıktı için seçmeli olarak yükseklik ve genişlik değerleri verilebilir"

```
$ python average.py ./images 170 240
```

If need a detailed explanation on how this works, head over to [Learn OpenCV](http://www.learnopencv.com/average-face-opencv-c-python-tutorial/).


## References

Some Papers

- C. Sagonas, E. Antonakos, G, Tzimiropoulos, S. Zafeiriou, M. Pantic. [300 faces In-the-wild challenge: Database and results](https://ibug.doc.ic.ac.uk/media/uploads/documents/sagonas_2016_imavis.pdf). Image and Vision Computing (IMAVIS), Special Issue on Facial Landmark Localisation "In-The-Wild". 2016.
- C. Sagonas, G. Tzimiropoulos, S. Zafeiriou, M. Pantic. [A semi-automatic methodology for facial landmark annotation](https://ibug.doc.ic.ac.uk/media/uploads/documents/sagonas_cvpr_2013_amfg_w.pdf). Proceedings of IEEE Int’l Conf. Computer Vision and Pattern Recognition (CVPR-W), 5th Workshop on Analysis and Modeling of Faces and Gestures (AMFG 2013). Oregon, USA, June 2013.
- C. Sagonas, G. Tzimiropoulos, S. Zafeiriou, M. Pantic. [300 Faces in-the-Wild Challenge: The first facial landmark localization Challenge](https://ibug.doc.ic.ac.uk/media/uploads/documents/sagonas_iccv_2013_300_w.pdf). Proceedings of IEEE Int’l Conf. on Computer Vision (ICCV-W), 300 Faces in-the-Wild Challenge (300-W). Sydney, Australia, December 2013.
