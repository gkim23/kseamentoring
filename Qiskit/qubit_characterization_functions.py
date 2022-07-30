import numpy as np
from scipy.optimize import curve_fit
from qiskit import pulse
from qiskit.circuit import Parameter
import matplotlib.pyplot as plt
from qiskit import IBMQ

IBMQ.load_account()
provider = IBMQ.get_provider(hub='ibm-q', group='open', project='main')
backend = provider.get_backend('ibmq_armonk')
backend_config = backend.configuration()
dt = backend_config.dt
def get_closest_multiple_of_16(num):
    return int(num + 8 ) - (int(num + 8 ) % 16)

def get_dt_from(sec):
    return get_closest_multiple_of_16(sec/dt)

def baseline_remove(values):
    return np.array(values) - np.mean(values)

def fit_function(x_values, y_values, function, init_params):
    fitparams, conv = curve_fit(function, x_values, y_values, init_params)
    y_fit = function(x_values, *fitparams)

    return fitparams, y_fit

def qubit_spectroscopy(backend, drive_duration_sec, drive_sigma_sec, drive_amp, qubit, mem_slot, frequency_list):
    drive_freq = Parameter('drive_freq')
    with pulse.build(backend=backend, default_alignment='sequential', name='Qubit Spectroscopy Experiment') as qubit_spec:
        drive_duration = get_closest_multiple_of_16(pulse.seconds_to_samples(drive_duration_sec))
        drive_sigma = pulse.seconds_to_samples(drive_sigma_sec)
        drive_chan = pulse.drive_channel(qubit)
        pulse.set_frequency(drive_freq, drive_chan)
        pulse.play(pulse.Gaussian(duration=drive_duration,
                                  amp=drive_amp,
                                  sigma=drive_sigma,
                                  name='spectroscopy pulse'), drive_chan)
        pulse.measure(qubits=[qubit], registers=[pulse.MemorySlot(mem_slot)])
    qubit_spec_schedule = [qubit_spec.assign_parameters({drive_freq: f}, inplace=False) for f in frequency_list]
    return qubit_spec_schedule

def qubit_spectroscopy_plot(qubit_spectro_job, qubit, scale_factor, frequency_list):
    spectro_results = qubit_spectro_job.result(timeout=120)
    spectro_values = []
    frequency_list = frequency_list/1e9
    for i in range(len(frequency_list)):
        spectro_values.append(spectro_results.get_memory(i)[qubit] * scale_factor)

    spectro_values = np.real(baseline_remove(spectro_values))

    fit_params, y_fit = fit_function(frequency_list,
                                     spectro_values,
                                     lambda x, A, q_freq, B, C: (A / np.pi) * (B / ((x - q_freq) ** 2 + B ** 2)) + C,
                                     [1, np.average(frequency_list), 1, -0.2])

    plt.scatter(frequency_list, spectro_values, color='black')
    plt.plot(frequency_list, y_fit, color='red')

    plt.xlabel("Drive Frequency [a.u.]", fontsize=15)
    plt.ylabel("Measured signal [a.u.]", fontsize=15)
    plt.show()
    return fit_params[1]*1e9

def rabi_schedule(backend, drive_duration_sec, drive_sigma_sec, qubit, qubit_frequency, mem_slot, amplitude_list):
    drive_amp = Parameter('drive_amp')
    with pulse.build(backend=backend, default_alignment='sequential', name='Rabi Experiment') as rabi_sched:
        drive_duration = get_closest_multiple_of_16(pulse.seconds_to_samples(drive_duration_sec))
        drive_sigma = pulse.seconds_to_samples(drive_sigma_sec)
        drive_chan = pulse.drive_channel(qubit)
        pulse.set_frequency(qubit_frequency, drive_chan)
        pulse.play(pulse.Gaussian(duration=drive_duration,
                                  amp=drive_amp,
                                  sigma=drive_sigma,
                                  name='Rabi Pulse'), drive_chan)
        pulse.measure(qubits=[qubit], registers=[pulse.MemorySlot(mem_slot)])

    rabi_schedules = [rabi_sched.assign_parameters({drive_amp: a}, inplace=False) for a in amplitude_list]
    return rabi_schedules

def rabi_plot(drive_amps, rabi_job, qubit, scale_factor):
    rabi_results = rabi_job.result(timeout=120)
    rabi_values = []
    for i in range(30):
        rabi_values.append(rabi_results.get_memory(i)[qubit] * scale_factor)

    rabi_values = np.real(baseline_remove(rabi_values))
    fit_params, y_fit = fit_function(drive_amps,
                                     rabi_values,
                                     lambda x, A, B, drive_period, phi: (
                                                 A * np.cos(2 * np.pi * x / drive_period - phi) + B),
                                     [3, 0.1, 0.3, 0])

    plt.scatter(drive_amps, rabi_values, color='black')
    plt.plot(drive_amps, y_fit, color='red')

    drive_period = fit_params[2]  # get period of rabi oscillation

    plt.axvline(drive_period / 2, color='red', linestyle='--')
    plt.axvline(drive_period, color='red', linestyle='--')
    plt.annotate("", xy=(drive_period, 0), xytext=(drive_period / 2, 0), arrowprops=dict(arrowstyle="<->", color='red'))
    plt.annotate("$\pi$", xy=(drive_period / 2 - 0.03, 0.1), color='red')

    plt.xlabel("Drive amp [a.u.]", fontsize=15)
    plt.ylabel("Measured signal [a.u.]", fontsize=15)
    plt.show()
    return drive_period * 0.5

def t1_schedule(backend, drive_duration_sec, drive_sigma_sec, qubit, pi_amp, qubit_frequency_ini, mem_slot, delay_times_sec):
    with pulse.build(backend) as pi_pulse:
        drive_duration = get_closest_multiple_of_16(pulse.seconds_to_samples(drive_duration_sec))
        drive_sigma = pulse.seconds_to_samples(drive_sigma_sec)
        drive_chan = pulse.drive_channel(qubit)
        pulse.play(pulse.Gaussian(duration=drive_duration,
                                  amp=pi_amp,
                                  sigma=drive_sigma,
                                  name='pi_pulse'), drive_chan)
    delay = Parameter('delay')
    with pulse.build(backend=backend, default_alignment='sequential', name="T1 delay Experiment") as t1_schedule:
        drive_chan = pulse.drive_channel(qubit)
        pulse.set_frequency(qubit_frequency_ini, drive_chan)
        pulse.call(pi_pulse)
        pulse.delay(delay, drive_chan)
        pulse.measure(qubits=[qubit], registers=[pulse.MemorySlot(mem_slot)])

    t1_schedules = [t1_schedule.assign_parameters({delay: get_dt_from(d)}, inplace=False) for d in delay_times_sec]
    return t1_schedules

def t1_measurement(t1_job, qubit, scale_factor, num_shots, delay_times_sec, us):
    t1_results = t1_job.result(timeout=120)

    t1_values = []

    for i in range(len(t1_results.results)):
        iq_data = t1_results.get_memory(i)[:,qubit] * scale_factor
        t1_values.append(-np.real(sum(iq_data) / num_shots))

    fit_params, y_fit = fit_function(delay_times_sec / us, t1_values,
                                     lambda x, A, C, T1: (A * np.exp(-x / T1) + C),
                                     [-3, 3, 100]
                                     )

    _, _, T1 = fit_params

    plt.scatter(delay_times_sec / us, t1_values, color='black')
    plt.plot(delay_times_sec / us, y_fit, color='red', label=f"T1 = {T1:.2f} us")
    plt.xlim(0, np.max(delay_times_sec / us))
    plt.title("$T_1$ Experiment", fontsize=15)
    plt.xlabel('Delay before measurement [$\mu$s]', fontsize=15)
    plt.ylabel('Signal [a.u.]', fontsize=15)
    plt.legend()
    plt.show()

def t2_schedule(backend, drive_duration_sec, drive_sigma_sec, qubit, drive_amp, qubit_frequency_ini, pi_pulse, mem_slot, delay_times_sec_t2):
    with pulse.build(backend) as pi2_pulse:
        drive_duration = get_closest_multiple_of_16(pulse.seconds_to_samples(drive_duration_sec))
        drive_sigma = pulse.seconds_to_samples(drive_sigma_sec)
        drive_chan = pulse.drive_channel(qubit)
        pulse.play(pulse.Gaussian(duration=drive_duration,
                                  amp=drive_amp,
                                  sigma=drive_sigma,
                                  name='pi2_pulse'), drive_chan)
    # %%
    delay = Parameter('delay')
    with pulse.build(backend=backend, default_alignment='sequential', name="T2 delay Experiment") as t2_schedule:
        drive_chan = pulse.drive_channel(qubit)
        pulse.set_frequency(qubit_frequency_ini, drive_chan)
        pulse.call(pi2_pulse)
        pulse.delay(delay, drive_chan)
        pulse.call(pi_pulse)
        pulse.delay(delay, drive_chan)
        pulse.call(pi2_pulse)
        pulse.measure(qubits=[qubit], registers=[pulse.MemorySlot(mem_slot)])

    t2_schedules = [t2_schedule.assign_parameters({delay: get_dt_from(d)}, inplace=False) for d in delay_times_sec_t2]

def t2_measurement(t2_echo_job, delay_times_sec_t2, qubit, scale_factor, num_shots_per_point, us):
    t2_results = t2_echo_job.result(timeout=120)

    t2_values = []

    for i in range(len(delay_times_sec_t2)):
        iq_data = t2_results.get_memory(i)[:, qubit] * scale_factor
        t2_values.append(-sum(iq_data) / num_shots_per_point)

    fit_params, y_fit = fit_function(delay_times_sec_t2 / us, t2_values,
                                     lambda x, A, C, T2: (A * np.exp(-x / T2) + C),
                                     [-3, 3, 100]
                                     )

    _, _, T2 = fit_params

    plt.scatter(delay_times_sec_t2 / us, t2_values, color='black')
    plt.plot(delay_times_sec_t2 / us, y_fit, color='red', label=f"T2 = {T2:.2f} us")
    plt.xlim(0, np.max(delay_times_sec_t2 / us))
    plt.title("$T_2$ Experiment", fontsize=15)
    plt.xlabel('Delay before measurement [$\mu$s]', fontsize=15)
    plt.ylabel('Signal [a.u.]', fontsize=15)
    plt.legend()
    plt.show()