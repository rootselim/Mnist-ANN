import numpy as np
import pandas as pd
from keras.datasets import mnist
from keras.utils import to_categorical #kategorik verilere cevirme
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras.models import Sequential #sirali model
from keras.layers import Dense #bagli katmanlar
from keras.models import load_model #modelin geri yuklenmesi
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')



#mnist veri setini yükle eğitim ve test veri seti olarak
(x_train,y_train),(x_test,y_test) = mnist.load_data()
plt.figure(figsize=(10,5))
for i in range(6):
    plt.subplot(2,3,i+1)
    plt.imshow(x_train[i],cmap="gray")
    plt.title(f"Label : {y_train[i]}")
    plt.axis('off')
# plt.show()

#veri setini normalize edelim 0-255 arasındaki px degerlerini 0 ve 1 arasına olceklendiriyor
x_train = x_train.reshape(60000,784).astype('float32')/255
x_test = x_test.reshape(10000,784).astype('float32')/255


y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

model = Sequential()
#ilk katman 512 cell Relu activation function input size = 28*28
model.add(Dense(512,activation="relu"))
#ikinci katman 256 cell activation: tanh
model.add(Dense(256,activation="tanh"))
#output layer 10 tane olmak zorunda cunku 10 adet y sınıfımız var,activation softmax (ikiden fazla sınıf varsa softmax kullanılmak zorunda)
model.add(Dense(10,activation="softmax"))
#model derlemesi : optimizer (adam: büyük veri ve kompleks aglar icin idealdir)
#model derlemesi : loss(categorical_crossentropy)
#metrik (accuracy)
model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=["accuracy"])
#Erken durdurma : eger validation loss iyilesmiyorsa egitimi durdurur.
early_stopping = EarlyStopping(monitor="val_loss", patience=3,restore_best_weights=True) # 3 epoch boyunca iyilesme olmazsa egitimi durdurur
#monitor : dogrulma setindeki kaybı izler
#patience : 3 -> 3 epoch boyunca val_los degismiyorsa erken durdurma yapar
#restore_best_weights : en iyi modelin ağırlıklarını geri yükler

#model checkpoint : en iyi modelin ağırlıklarını kaydeder
checkpoint = ModelCheckpoint("ann_best_model.keras", monitor="val_loss", save_best_only=True)
#save_best_only: sadece en iyi performans gosteren modeli kaydeder
#Model Training : 10 epochs ,batch_size : 64, dogrulama seti oranı : 0.20
#model 60000 veri setini her biri 60 parcadan olusan 1000 kerede train edecek ve biz buna 1 epoch diyecegiz ama
# validation ayrımı yapıldıgı icin 60000 train verisi degil de 48000 train verisi var bu nedenle her biri 60 parcadan 8000 kerede train yapılacak

model.fit(x_train,y_train
          ,batch_size=60 #60lı parcalar ile egitim yapılacak
          ,epochs=10 #model toplamda 10 kere veri setini gorecek.Veri seti 10 kere egitilecek
          ,validation_split=0.2 # test seti %20 olacak
          ,callbacks=[early_stopping,checkpoint])

#test verisi ile model performansı degerlendirme
#evaluate = modelin test verisi üzerindeki loss ve accuracy degerini hesaplar
test_loss, test_acc = model.evaluate(x_test,y_test)
#print(test_acc) #test acc 0.97 yani yuzde 97 ihtimalle dogru tahmin yapılıyor

model.save("ann_best_model.h5")
loaded_model = load_model("ann_best_model.h5")
test_loss2,test_acc2 = loaded_model.evaluate(x_test,y_test)
print(test_loss2,test_acc2)












