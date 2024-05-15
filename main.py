import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit.utils import QuantumInstance
from qiskit_machine_learning.algorithms import VQC
from qiskit_machine_learning.kernels import QuantumKernel
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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

# Normalize the features to the range [0, 1]
normalizer = MinMaxScaler()
X_train = normalizer.fit_transform(X_train)
X_test = normalizer.transform(X_test)

# Set up the feature map and variational form
num_qubits = 4
feature_map = ZZFeatureMap(feature_dimension=num_qubits, reps=2, entanglement='linear')
var_form = RealAmplitudes(num_qubits, reps=3)

# Set up the quantum instance
quantum_instance = QuantumInstance(backend='qasm_simulator', shots=1024)

# Set up the VQC
vqc = VQC(feature_map=feature_map, ansatz=var_form, optimizer='COBYLA', warm_start=True)

# Train the VQC
vqc.fit(X_train, y_train)

# Make predictions on the test set
y_pred = vqc.predict(X_test)

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')

print("Evaluation Metrics:")
print(f"Accuracy: {accuracy:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")
print(f"F1-score: {f1:.3f}")

# Use QuantumKernel for classification
quantum_kernel = QuantumKernel(feature_map=feature_map, quantum_instance=quantum_instance)
vqc_kernel = VQC(feature_map=quantum_kernel, ansatz=var_form, optimizer='COBYLA', warm_start=True)
vqc_kernel.fit(X_train, y_train)
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