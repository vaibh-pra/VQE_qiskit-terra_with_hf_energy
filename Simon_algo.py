from qiskit import QuantumCircuit, Aer, execute, QuantumRegister, ClassicalRegister, assemble
from qiskit.tools.visualization import plot_histogram
from qiskit_textbook.tools import simon_oracle
import numpy as np
from sympy import *
%matplotlib inline

# Define the black-box function f
f = '111'



# Create the circuit
n = len(f)
qreg = QuantumRegister(2*n)
creg = ClassicalRegister(n)
qc = QuantumCircuit(qreg, creg)

#Apply Hadmarad to the first n qubits
qc.h(range(n))

qc.barrier()

# Apply the black-box function f
qc2 = simon_oracle(f)
qc.compose(qc2, inplace=True)

qc.barrier()

#Apply Hadamard to the first n qubits
qc.h(range(n))

#Measure first n qubits
qc.measure(range(n), range(n))

# Simulate the circuit
backend = Aer.get_backend('qasm_simulator')
job = assemble(qc, shots=500)
result = backend.run(job).result()

# Extract the result
cts = result.get_counts()
del cts['000']

#
b = np.zeros(len(cts))
coeff = []
for key in cts:
    key_str = str(key)
    z = key_str[::-1]
    row = []
    for i in range(len(z)):
        row.append(int(z[i]))
    coeff.append(row)
print(np.array(coeff), b)
A = np.array(coeff)
x = np.linalg.solve(A, b)
print(x)
plot_histogram(cts)
