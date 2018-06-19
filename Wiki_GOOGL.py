import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import linear_model
import quandl

# считываем данные из файла
data = quandl.get('Wiki/GOOGL')
x = data[['Open', 'High', 'Low']]
y = data.Close

# делим датасет на данные для обучения и данные для проверки
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42, shuffle = True)

# точки необходимо отсортировать для корректного отображения графика
y_test = y_test.sort_index()
x_test = x_test.sort_index()

# вычисляем линейную регрессию
regr = linear_model.LinearRegression()
regr.fit(x_train, y_train)
y_pred = regr.predict(x_test)

# строим график
plt.scatter(y_test.index, y_test,  color='red', alpha = 0.5)
plt.plot(y_test.index, y_pred, color='blue', linewidth=1)
plt.xlabel("Date")
plt.ylabel("Close")
plt.show()
