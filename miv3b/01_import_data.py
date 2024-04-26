# %%
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# The above lines import necessary libraries.
# matplotlib.pyplot is used for plotting data.
# train_test_split from sklearn.model_selection is used to split the data into training and testing sets.

# %%
# Wczytanie danych z pliku
with open("dane15.txt", "r") as file:
    data = file.readlines()

# The above code reads data from a file named 'dane15.txt' located in the 'Dane' directory.
# Each line in the file is read and stored in the 'data' variable as a list of strings.

# %%
# Podział danych na dane treningowe i testowe
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# The above code splits the data into training and testing sets.
# 80% of the data is used for training (train_data) and 20% for testing (test_data).
# The split is random, but we set 'random_state' to 42 to make the results reproducible.

# %%
# Wykres prezentujący punkty z danych
x_values = []
y_values = []

for line in data:
    x, y = map(float, line.split())
    # Zakładamy, że dane są oddzielone spacją
    x_values.append(x)
    y_values.append(y)

# The above code creates two empty lists, 'x_values' and 'y_values'.
# Then, it iterates over each line in the data.
# Each line is split into 'x' and 'y' values, assuming they are separated by a space.
# The 'x' and 'y' values are converted from strings to floats and then added to the 'x_values' and 'y_values' lists, respectively.

# %%
# Wykres
plt.scatter(x_values, y_values, color='blue', label='Dane')
plt.xlabel('Wartość X')
plt.ylabel('Wartość Y')
plt.title('Wykres punktów danych')
plt.legend()
plt.show()

# The above code creates a scatter plot of the data points.
# The 'x_values' list is used for the x-axis and the 'y_values' list for the y-axis.
# The color of the points is set to blue and they are labeled as 'Dane'.
# The x-axis and y-axis are labeled as 'Wartość X' and 'Wartość Y', respectively.
# The plot is given a title 'Wykres punktów danych'.
# A legend is added to the plot.
# Finally, the plot is displayed.
