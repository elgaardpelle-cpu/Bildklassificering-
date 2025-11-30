import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

digits = load_digits()
X = digits.data
y = digits.target

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42
)

model = SVC(gamma=0.001, C=10)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Modellens noggrannhet:", accuracy)

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, cmap="Blues", fmt="d",
            xticklabels=digits.target_names,
            yticklabels=digits.target_names)
plt.title("Confusion Matrix")
plt.xlabel("Prediktion")
plt.ylabel("Sant värde")
plt.tight_layout()
plt.show(block=False)
plt.pause(5)
plt.close()

print("\nKlassificeringsrapport:\n", classification_report(y_test, y_pred))

scores = cross_val_score(model, X_scaled, y, cv=5)
print("Cross-validation noggrannhet per fold:", scores)
print("Genomsnittlig CV-noggrannhet:", scores.mean())

fig, axes = plt.subplots(2, 5, figsize=(10,5))
for ax, image, label, pred in zip(axes.flatten(), X_test[:10], y_test[:10], y_pred[:10]):
    ax.imshow(image.reshape(8,8), cmap="gray")
    ax.set_title(f"Rätt: {label}, Pred: {pred}")
    ax.axis("off")
plt.tight_layout()
plt.show(block=False)
plt.pause(6)
plt.close()
