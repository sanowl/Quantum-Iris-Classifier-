import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit.utils import QuantumInstance
from qiskit_machine_learning.algorithms import VQC
from qiskit_machine_learning.kernels import QuantumKernel
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.svm import SVC

# Load and preprocess the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Set up the feature map and variational form
num_qubits = 6
feature_map = ZZFeatureMap(feature_dimension=num_qubits, reps=3, entanglement='full')
var_form = RealAmplitudes(num_qubits, reps=4)

# Set up the quantum instance
quantum_instance = QuantumInstance(backend='qasm_simulator', shots=1024)

# Set up the VQC
vqc = VQC(
    feature_map=feature_map,
    ansatz=var_form,
    optimizer=GridSearchCV(
        estimator=VQC(
            feature_map=feature_map,
            ansatz=var_form,
            optimizer='COBYLA',
            warm_start=True
        ),
        param_grid={'optimizer__maxiter': [100, 200, 300]},
        cv=3,
        n_jobs=-1,
        verbose=2
    ),
    warm_start=True
)

# Train the VQC
vqc.fit(X_train, y_train)

# Make predictions on the test set
y_pred = vqc.predict(X_test)

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')

print("Evaluation Metrics (VQC):")
print(f"Accuracy: {accuracy:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")
print(f"F1-score: {f1:.3f}")

# Use QuantumKernel for classification
quantum_kernel = QuantumKernel(feature_map=feature_map, quantum_instance=quantum_instance)
vqc_kernel = VQC(
    feature_map=quantum_kernel,
    ansatz=var_form,
    optimizer=GridSearchCV(
        estimator=VQC(
            feature_map=quantum_kernel,
            ansatz=var_form,
            optimizer='COBYLA',
            warm_start=True
        ),
        param_grid={'optimizer__maxiter': [100, 200, 300]},
        cv=3,
        n_jobs=-1,
        verbose=2
    ),
    warm_start=True
)
vqc_kernel.fit(X_train, y_train)

# Make predictions on the test set using QuantumKernel
y_pred_kernel = vqc_kernel.predict(X_test)

# Calculate evaluation metrics for QuantumKernel
accuracy_kernel = accuracy_score(y_test, y_pred_kernel)
precision_kernel = precision_score(y_test, y_pred_kernel, average='macro')
recall_kernel = recall_score(y_test, y_pred_kernel, average='macro')
f1_kernel = f1_score(y_test, y_pred_kernel, average='macro')

print("\nEvaluation Metrics (QuantumKernel):")
print(f"Accuracy: {accuracy_kernel:.3f}")
print(f"Precision: {precision_kernel:.3f}")
print(f"Recall: {recall_kernel:.3f}")
print(f"F1-score: {f1_kernel:.3f}")

# Train and evaluate a classical SVM classifier
svm = SVC(kernel='rbf', C=1.0, random_state=42)
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)

# Calculate evaluation metrics for SVM
accuracy_svm = accuracy_score(y_test, y_pred_svm)
precision_svm = precision_score(y_test, y_pred_svm, average='macro')
recall_svm = recall_score(y_test, y_pred_svm, average='macro')
f1_svm = f1_score(y_test, y_pred_svm, average='macro')

print("\nEvaluation Metrics (SVM):")
print(f"Accuracy: {accuracy_svm:.3f}")
print(f