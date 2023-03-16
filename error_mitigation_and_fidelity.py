#!/usr/bin/env python
# coding: utf-8

# In[157]:


from qiskit import execute
from qiskit import BasicAer, Aer
from qiskit import IBMQ, QuantumCircuit
from qiskit.providers.fake_provider import FakeManila
from qiskit.providers.aer.noise import NoiseModel
from qiskit.utils import QuantumInstance
from qiskit.ignis.mitigation.measurement import CompleteMeasFitter, complete_meas_cal
from qiskit.tools.visualization import plot_histogram, plot_state_city
from qiskit.quantum_info import partial_trace, state_fidelity
import numpy as np
import qiskit.tools.jupyter
get_ipython().run_line_magic('qiskit_job_watcher', '')


# In[171]:


bell = QuantumCircuit(2,2)
bell.x(0)
bell.h(0)
bell.rx(np.pi/2, 1)
bell.cx(0,1)
bell.measure([0,1],[0,1])
bell.draw(output = 'mpl')


# In[81]:


IBMQ.load_account()


# In[172]:


backend1 = Aer.get_backend('aer_simulator')
device = FakeManila()
noise_model = NoiseModel.from_backend(device)
backend2 = Aer.get_backend('qasm_simulator')


# In[173]:


res1 = execute(bell, backend = backend1, shots = 1024, noise_model = noise_model).result()
raw_counts = res1.get_counts()
res2 = execute(bell, backend = backend2, shots =1024).result()
sim_counts = res2.get_counts()


# In[174]:


plot_histogram([raw_counts, sim_counts], legend=['noisy','sim'])


# In[175]:


cal_bell, state_labels = complete_meas_cal(qr = bell.qregs[0], circlabel = 'measerrormitigationcal')
cal_bell[1].draw(output='mpl')


# In[176]:


cal_result = execute(cal_bell, backend = backend2, shots = 1024, noise_model = noise_model).result()
cal_counts = cal_result.get_counts(cal_bell[1])
plot_histogram(cal_counts)


# In[177]:


meas_fitter = CompleteMeasFitter(cal_result, state_labels)
meas_fitter.plot_calibration()


# In[178]:


meas_filter = meas_fitter.filter
res3 = meas_filter.apply(res1)
mitigated_counts = res3.get_counts()
plot_histogram([raw_counts, mitigated_counts], legend=['noisy', 'mitigated'])


# In[179]:


res4 = execute(bell, backend = Aer.get_backend('statevector_simulator')).result()
raw_state_vec = res4.get_statevector()
expected_statevector = [0, 1/np.sqrt(2), 1/np.sqrt(2), 0]
print(state_fidelity(raw_state_vec, expected_statevector))


# In[ ]:




