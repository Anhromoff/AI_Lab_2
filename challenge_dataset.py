import pandas
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import linear_model

# считываем данные из файла
data = pandas.read_csv('challenge_dataset.txt', names = ['x', 'y'])
x = data['x']
y = data['y']

# строим точки
plt.scatter(x, y,  color='red', alpha = 0.5)

# делим датасет на данные для обучения и данные для проверки
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=50)

# вычисляем линейную регрессию
regr = linear_model.LinearRegression()
regr.fit(x_train.values.reshape(-1, 1), y_train)
y_pred = regr.predict(x_test.values.reshape(-1, 1))

# строим график прямой
plt.plot(x_test, y_pred, color='blue', linewidth=2)
plt.xlabel("x")
plt.ylabel("y")
plt.show()