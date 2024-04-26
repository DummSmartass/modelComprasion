# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# The above lines import necessary libraries.
# numpy is used for numerical operations.
# matplotlib.pyplot is used for plotting data.
# train_test_split from sklearn.model_selection is used to split the data into training and testing sets.
# r2_score from sklearn.metrics is used to evaluate the performance of the model.

# %%
# Wczytanie danych z pliku
with open("dane1.txt", "r") as file:
    data = file.readlines()

# The above code reads data from a file named 'dane1.txt' located in the 'Dane' directory.
# Each line in the file is read and stored in the 'data' variable as a list of strings.

# %%
# Przetwarzanie danych do postaci potrzebnej do dopasowania modelu
x_data = []
y_data = []

for line in data:
    x, y = map(float, line.split())  # Zakładamy, że dane są oddzielone spacją
    x_data.append(x)
    y_data.append(y)

# The above code creates two empty lists, 'x_data' and 'y_data'.
# Then, it iterates over each line in the data.
# Each line is split into 'x' and 'y' values, assuming they are separated by a space.
# The 'x' and 'y' values are converted from strings to floats and then added to the 'x_data' and 'y_data' lists, respectively.

# %%
# Podział danych na dane treningowe i testowe
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42)

# The above code splits the data into training and testing sets.
# 80% of the data is used for training (x_train, y_train) and 20% for testing (x_test, y_test).
# The split is random, but we set 'random_state' to 42 to make the results reproducible.

# %%
# Dopasowanie modelu liniowego do danych treningowych
X_train = np.array(x_train).reshape(-1, 1)  # Przekształcamy x_train do postaci kolumnowej
Y_train = np.array(y_train)

# The above code converts the 'x_train' and 'y_train' lists to numpy arrays.
# The 'x_train' array is reshaped to a column vector.

# Obliczenie parametrów modelu (a i b) za pomocą metody najmniejszych kwadratów
# Funkcja np.vstack w Pythonie służy do łączenia tablic NumPy wzdłuż osi pionowej, czyli wierszy.
# Funkcja np.linalg.lstsq w bibliotece NumPy służy do rozwiązywania nadokreślonych układów równań liniowych w najmniejszych kwadratach
# rcond jest parametrem, który określa poziom tolerancji dla wyznaczenia odwrotności macierzy
# rcond=None w funkcji np.linalg.lstsq oznacza, że wartość rcond nie jest jawnie określona, co oznacza, że zostanie użyta domyślna wartość.
A = np.vstack([X_train.T, np.ones(len(X_train))]).T
a, b = np.linalg.lstsq(A, Y_train, rcond=None)[0]

# The above code uses the least squares method to find the optimal parameters (a and b) for the linear model.
# The 'A' matrix is created by adding a row of ones to the transpose of the 'X_train' matrix.
# The transpose of 'A' is then taken to get the correct shape.
# The 'a' and 'b' variables store the optimal parameters.

# %%
# Ocena modelu na danych testowych
X_test = np.array(x_test).reshape(-1, 1)
Y_test = np.array(y_test)
y_pred = a*X_test + b
precision = r2_score(Y_test, y_pred)

# The above code converts the 'x_test' and 'y_test' lists to numpy arrays.
# The 'x_test' array is reshaped to a column vector.
# The linear model is used to make predictions on the test data.
# The r2_score function is used to evaluate the performance of the model on the test data.

# %%
# Wykres prezentujący punkty z danych i dopasowany model liniowy
plt.scatter(x_train, y_train, color='blue', label='Dane treningowe')
plt.scatter(x_test, y_test, color='green', label='Dane testowe')
plt.plot(X_train, a*X_train + b, color='red', label=f'Model liniowy: y = {a:.2f}x + {b:.2f}')
plt.xlabel('Wartość X')
plt.ylabel('Wartość Y')
plt.title(f'r2_score modelu na danych testowych: {precision:.2f}')
plt.legend()
plt.show()

# The above code creates a scatter plot of the training and testing data points.
# The linear model is plotted as a red line.
# The x-axis and y-axis are labeled as 'Wartość X' and 'Wartość Y', respectively.
# The plot is given a title showing the r2_score of the model on the test data.
# A legend is added to the plot.
# Finally, the plot is displayed.
