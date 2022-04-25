from qiskit.tools.jupyter import *
from qiskit import IBMQ
import numpy as np
from qiskit.tools.monitor import job_monitor
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from qiskit import pulse
from qiskit.circuit import Parameter
from qubit_characterization_functions import *

IBMQ.load_account()
provider = IBMQ.get_provider(hub='ibm-q', group='open', project='main')
backend = provider.get_backend('ibmq_armonk')
backend_config = backend.configuration()
dt = backend_config.dt
backend_defaults = backend.defaults()
qubit = 0
mem_slot = 0
qubit_frequency_ini = backend_defaults.qubit_freq_est[qubit]

GHz = 1.0e9 # Gigahertz
MHz = 1.0e6 # Megahertz
us = 1.0e-6 # Microseconds
ns = 1.0e-9 # Nanoseconds

scale_factor = 1e-14

drive_sigma_sec = 0.075 * us
drive_duration_sec = drive_sigma_sec * 8

frequency_list = np.arange(qubit_frequency_ini - 20*MHz, qubit_frequency_ini + 20*MHz, 1*MHz)
qubit_spec_schedule = qubit_spectroscopy(backend=backend, drive_duration_sec=drive_duration_sec, drive_sigma_sec=drive_sigma_sec,
                                         drive_amp=0.05, qubit=qubit, mem_slot=mem_slot, frequency_list=frequency_list)
#%%
num_shots_per_point = 1024

qubit_spectro_job = backend.retrieve_job('6248fd49bcb717581bb24bc9')

job_monitor(qubit_spectro_job)
#%%
qubit_frequency_meas = qubit_spectroscopy_plot(qubit_spectro_job = qubit_spectro_job, qubit = qubit, scale_factor = scale_factor, frequency_list = frequency_list)


#%%
amplitude_min = 0
amplitude_max = 0.8
amplitude_list = np.linspace(amplitude_min, amplitude_max, 30)
#%%
rabi_schedules = rabi_schedule(backend = backend, drive_duration_sec = drive_duration_sec, drive_sigma_sec = drive_sigma_sec,
                               qubit = qubit, qubit_frequency = qubit_frequency_meas, mem_slot = mem_slot, amplitude_list = amplitude_list)
#%%
num_shots_per_point = 1024
rabi_job = backend.run(rabi_schedules,
                  meas_level=1,
                  meas_return='avg',
                  shots=num_shots_per_point)
job_monitor(rabi_job)
#%%
Xgate_amplitude = rabi_plot(drive_amps = amplitude_list, rabi_job = rabi_job, qubit = qubit, scale_factor = scale_factor)

pi_amp = abs(Xgate_amplitude)
time_max_sec = 450 * us
time_step_sec = 6.5 * us
delay_times_sec = np.arange(1 * us, time_max_sec, time_step_sec)

t1_schedules = t1_schedule(backend = backend, drive_duration_sec = drive_duration_sec,
                           drive_sigma_sec = drive_sigma_sec, qubit = qubit, pi_amp = pi_amp, qubit_frequency_ini =
                           qubit_frequency_ini, mem_slot = mem_slot, delay_times_sec = delay_times_sec)

num_shots = 256
t1_job = backend.run(rabi_schedules, meas_level = 1, meas_return = 'avg', shots = num_shots_per_point)
job_monitor(t1_job)

t1_measurements = t1_measurement(qubit = qubit, scale_factor = scale_factor, num_shots = num_shots, delay_times_sec
                                  = delay_times_sec, us = us)