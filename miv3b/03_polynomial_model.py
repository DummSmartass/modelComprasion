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

# The above code reads data from a file named 'dane1.txt'.
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
# Przetwarzanie danych treningowych do postaci potrzebnej do dopasowania modelu
X_train = np.array(x_train).reshape(-1, 1)  # Przekształcamy x_train do postaci kolumnowej
Y_train = np.array(y_train)

# The above code converts the 'x_train' and 'y_train' lists to numpy arrays.
# The 'x_train' array is reshaped to a column vector.

# Dodanie kolumny x^2 do danych treningowych
# Funkcja np.hstack w bibliotece NumPy służy do łączenia tablic NumPy wzdłuż osi poziomej, czyli kolumn.
X_train_poly = np.hstack([X_train, X_train**2])

# The above code adds a new column to the 'X_train' array, where each element in the new column is the square of the corresponding element in the original column.

# %%
# Obliczenie parametrów modelu wielomianowego (a0, a1, a2) za pomocą metody najmniejszych kwadratów
# Funkcja np.linalg.lstsq w bibliotece NumPy służy do rozwiązywania nadokreślonych układów równań liniowych w najmniejszych kwadratach
A = np.hstack([X_train_poly, np.ones((X_train_poly.shape[0], 1))])
params = np.linalg.lstsq(A, Y_train, rcond=None)[0]

# The above code uses the least squares method to find the optimal parameters (a0, a1, a2) for the polynomial model.
# The 'A' matrix is created by adding a column of ones to the 'X_train_poly' matrix.
# The 'params' variable stores the optimal parameters.

# %%
# Przetwarzanie danych testowych
X_test = np.array(x_test).reshape(-1, 1)
Y_test = np.array(y_test)
X_test_poly = np.hstack([X_test, X_test**2])
y_pred = np.dot(X_test_poly, params[:2]) + params[2]  # Obliczenie predykcji modelu wielomianowego
precision = r2_score(Y_test, y_pred)

# The above code converts the 'x_test' and 'y_test' lists to numpy arrays.
# The 'x_test' array is reshaped to a column vector.
# A new column is added to the 'X_test' array, where each element in the new column is the square of the corresponding element in the original column.
# The polynomial model is used to make predictions on the test data.
# The r2_score function is used to evaluate the performance of the model on the test data.

# %%
# Wykres prezentujący punkty z danych i dopasowany model wielomianowy
plt.scatter(x_train, y_train, color='blue', label='Dane treningowe')
plt.scatter(x_test, y_test, color='green', label='Dane testowe')
x_range = np.linspace(min(x_train), max(x_train), 100)
plt.plot(x_range, params[0]*x_range + params[1]*x_range**2 + params[2], color='red', label=f'Model wielomianowy: y = {params[0]:.2f}x^2 + {params[1]:.2f}x + {params[2]:.2f}')
plt.xlabel('Wartość X')
plt.ylabel('Wartość Y')
plt.title(f'r2_score modelu na danych testowych: {precision:.2f}')
plt.legend()
plt.show()

# The above code creates a scatter plot of the training and testing data points.
# The polynomial model is plotted as a red line.
# The x-axis and y-axis are labeled as 'Wartość X' and 'Wartość Y', respectively.
# The plot is given a title showing the r2_score of the model on the test data.
# A legend is added to the plot.
# Finally, the plot is displayed.
