import pandas as pd # bu kütüphane ile veri yüklenir. dataframe'i(excel dosyası) okunur verilerin.

# Veriyi yükleme
data_path = 'housing.csv'
data = pd.read_csv(data_path)

# İlk birkaç satırı gösterme
print(data.head())
print(data.info()) #genel bilgiler


# Eksik değerleri kontrol etme
print(data.isnull().sum()) #her sütunda kaç tane eksik değer olduğu kontrol edilir

# Eksik değerleri silme işlemi yapılır.
data = data.dropna()


from sklearn.preprocessing import LabelEncoder

# Kategorik verileri etiketleme
# LabelEncoder kullanılarak ocean_proximity sütunundaki kategorik değerler sayısal değerlere dönüştürülür.Model eğitimine uygun hale getirilir.
label_encoder = LabelEncoder()
data['ocean_proximity'] = label_encoder.fit_transform(data['ocean_proximity'])


from sklearn.model_selection import train_test_split

# Özellikler ve hedef değişkeni ayırma
# median_house_value hedef değişken olarak ayrılır, geri kalan sütunlar (özellikler) X değişkenine atanır.
X = data.drop("median_house_value", axis=1)
y = data["median_house_value"]

# Eğitim ve test verisi olarak ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#veri setinin %20 si test için ayrılır

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Modeli oluşturma
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
# n_estimators=100 adet karar ağacı kullanılır

# Modeli eğitme
rf_model.fit(X_train, y_train)

# Test seti üzerinde tahmin yapma
y_pred = rf_model.predict(X_test)

# Model performansını değerlendirmede iki yöntem vardır. ki kare ve Mse
mse = mean_squared_error(y_test, y_pred) # test setindeki gerçek değerler (y_test), tahmin edilen değerler (y_pred)
rmse = mse ** 0.5
print(f"Karekök ortalama hata: {rmse}")


import pickle

# Modeli kaydetme
model_path = 'random_forest_model.pkl'
with open(model_path, 'wb') as file:
    pickle.dump(rf_model, file)
