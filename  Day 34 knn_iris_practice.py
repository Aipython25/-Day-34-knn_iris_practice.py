# Step 1: Libraries import करो
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Step 2: Dataset load करो
iris = load_iris()
X = iris.data       # features
y = iris.target     # labels (Setosa, etc.)

# Step 3: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Step 4: Model बनाओ
model = KNeighborsClassifier(n_neighbors=3)  # K=3 लिया है
model.fit(X_train, y_train)

# Step 5: Prediction करो
y_pred = model.predict(X_test)

# Step 6: Accuracy चेक करो
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# practice 1

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# अलग-अलग k के लिए accuracy check करो
for k in range(1, 11):
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"K = {k} → Accuracy = {acc}")
# practice 2 

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

# New flower: sepal_length, sepal_width, petal_length, petal_width
new_flower = [[5.1, 3.5, 1.4, 0.2]]
prediction = model.predict(new_flower)

print("Predicted Class Index:", prediction[0])
print("Predicted Class Name:", iris.target_names[prediction[0]])
# practice 3

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

cm = confusion_matrix(y_test, y_pred)

# Confusion Matrix Plot
sns.heatmap(cm, annot=True, cmap="Blues", xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()