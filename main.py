from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit.utils import QuantumInstance
from qiskit.algorithms import VQC
from qiskit.algorithms.optimizers import SPSA
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

num_qubits = 2
feature_map = ZZFeatureMap(feature_dimension=num_qubits, reps=2, entanglement='linear')
var_form = RealAmplitudes(num_qubits, reps=3)

# Set up the variational quantum classifier
vqc = VQC(feature_map=feature_map, ansatz=var_form, optimizer=SPSA(maxiter=100),
          quantum_instance=QuantumInstance(Aer.get_backend('qasm_simulator'), shots=1024))

vqc.fit(X_train, y_train)

# Evaluate the classifier on the test set
y_pred = vqc.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.3f}")