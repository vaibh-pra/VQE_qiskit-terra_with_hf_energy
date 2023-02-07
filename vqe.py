import numpy as np
import pylab
import matplotlib.pyplot as plt
import copy
from qiskit import BasicAer, Aer
from qiskit_nature.drivers import Molecule
from qiskit_nature.settings import settings
settings.dict_aux_operators = False
from qiskit.utils import algorithm_globals, QuantumInstance
from qiskit.algorithms import minimum_eigen_solvers, VQE
from qiskit.algorithms.minimum_eigensolvers import MinimumEigensolver
from qiskit_nature.circuit.library import HartreeFock, UCCSD
#from qiskit_nature.components.variational_forms import UCCSD
from qiskit_nature.second_q.drivers.pyscfd import pyscfdriver
from qiskit_nature.drivers.second_quantization import (
    ElectronicStructureMoleculeDriver, ElectronicStructureDriverType)
from qiskit_nature.problems.second_quantization import ElectronicStructureProblem
from qiskit_nature.mappers.second_quantization import ParityMapper
from qiskit_nature.transformers.second_quantization.electronic import FreezeCoreTransformer
from qiskit_nature.converters.second_quantization import QubitConverter
from qiskit.opflow import TwoQubitReduction
from qiskit_nature.algorithms import (GroundStateEigensolver,
                                      NumPyMinimumEigensolverFactory)
from qiskit.algorithms.optimizers import SLSQP, COBYLA
from qiskit import IBMQ
IBMQ.load_account()
import qiskit.tools.jupyter
from qiskit.circuit.library import EfficientSU2
from qiskit.providers.fake_provider import FakeManila
from qiskit.providers.aer.noise import NoiseModel
from qiskit.utils.mitigation import CompleteMeasFitter
%qiskit_job_watcher
distances = np.arange(0.5, 3.0, 0.1)
vqe_energies = []
hf_energies = []
exact_energies = []
backend0 = BasicAer.get_backend("statevector_simulator")
provider = IBMQ.get_provider(hub='ibm-q', group='open', project='main')
backend1 = provider.get_backend('ibmq_quito')
backend2 = Aer.get_backend('aer_simulator')
device = FakeManila()
NOISE_MODEL = NoiseModel.from_backend(device)
coupling_map = device.configuration().coupling_map
qi = QuantumInstance(backend = backend2, noise_model = NOISE_MODEL, coupling_map = coupling_map,
                     measurement_error_mitigation_cls = CompleteMeasFitter)
optimizer = COBYLA(maxiter = 1000)


for i,d in enumerate(distances):
    print('step', i)
    
    #setup the experiment
    
    
    molecule = Molecule(geometry=[["H", [0.0, 0.0, 0.0]],["H", [d, 0.0, 0.0]]],
        multiplicity=1,  # = 2*spin + 1
        charge=0)
    
    driver = ElectronicStructureMoleculeDriver(molecule=molecule, basis="sto6g", 
                                               driver_type=ElectronicStructureDriverType.PYSCF)
    qmolecule = driver.run()
    num_particles = (qmolecule
                        .get_property("ParticleNumber")
                        .num_particles)
    num_spin_orbitals = int(qmolecule
                            .get_property("ParticleNumber")
                            .num_spin_orbitals)
    
    
    
    problem = ElectronicStructureProblem(driver, [FreezeCoreTransformer(freeze_core=True)])

    operator = problem.second_q_ops()
    num_spin_orbitals = problem.num_spin_orbitals
    num_particles = problem.num_particles
    
    mapper = ParityMapper()
    hamiltonian = operator[0]
    converter = QubitConverter(mapper,two_qubit_reduction=True)
    reducer = TwoQubitReduction(num_particles)
    qubit_op = converter.convert(hamiltonian)
    qubit_op = reducer.convert(qubit_op)

    
   

    #Exact Result
    
    solver = NumPyMinimumEigensolverFactory()
    calc = GroundStateEigensolver(converter, solver)
    result = calc.solve(problem)
    hf_energies.append(result.hartree_fock_energy)
    exact_energies.append(result.total_energies[0].real)
#    print(hf_energies)
#    print(exact_energies)
    
    
    
    #VQE
    
    initial_state = HartreeFock(num_spin_orbitals, num_particles, converter)
    ansatz = UCCSD(converter,
                     num_particles,
                     num_spin_orbitals,
                     initial_state=initial_state)
    
    ansatz2 = EfficientSU2(qubit_op.num_qubits, entanglement="linear")
    
    vqe = VQE(ansatz, optimizer, quantum_instance = backend0)
    vqe_calc = vqe.compute_minimum_eigenvalue(qubit_op)
    vqe_result = problem.interpret(vqe_calc).total_energies[0].real
    vqe_energies.append(vqe_result)
#    print(vqe_energies)
    
    
plt.plot(distances, exact_energies, label = 'Exact Energy')
plt.plot(distances, vqe_energies, label = 'VQE Energy')
plt.plot(distances, hf_energies, label = 'HF Energy')
plt.legend()
plt.show()
