import matplotlib.pyplot as plt
import pylab as plab
from mpl_toolkits.mplot3d import Axes3D
import qutip as qt
import matplotlib.animation as animation
import numpy as np
from scipy.interpolate import CubicSpline


def plot_bloch_vector_component(tlist, bloch_array):
    fig, ax = plt.subplots(figsize=(12,5))
    ax.plot(tlist, bloch_array[0])
    ax.plot(tlist, bloch_array[1])
    ax.plot(tlist, bloch_array[2])
    ax.legend(("x","y","z"))
    ax.set_title("Bloch Sphere Component")
    plt.xlabel("Time[ns]")


def animate_bloch(vector_array, name="bloch.mp4", fps_in=20):
    fig = plab.figure()
    ax = Axes3D(fig,azim=-40,elev=30)
    sphere = qt.Bloch(axes=ax)
    points = [vector_array[0],vector_array[1],vector_array[2]]

    def ani(i):
        sx, sy, sz = points
        sphere.clear()
        sphere.add_vectors([sx[i],sy[i],sz[i]])
        sphere.make_sphere()
        return ax

    def init():
        return ax

    ani = animation.FuncAnimation(fig, ani, np.arange(len(vector_array[0])), init_func=init, blit=False, repeat=False)
    ani.save(name, fps=fps_in)



def driven_hamiltonian(omega, free_hamiltonian, qubit_x, qubit_y, plot=False):
    tlist = np.linspace(0,len(omega),len(omega))
    qubit_I = np.real(omega) * 0.9582815089328248 / 2
    qubit_Q = np.imag(omega) * 0.9582815089328248 / 2

    H_drive = qubit_x, qubit_y

    pulse_array = np.array([qubit_I, qubit_Q])

    def make_interpolation(id):
        def _function(t,args=None):
            return CubicSpline(tlist,pulse_array[id], bc_type="clamped")(t)
        return _function

    H0 = free_hamiltonian
    Ht_list = []
    Ht_list.append(H0)

    for ii in range(2):
        Ht_list.append([H_drive[ii], make_interpolation(ii)])

    if plot:
        fig, ax = plt.subplots(1,1,figsize=(12,10))
        ax.plot(tlist, Ht_list[1][1](tlist) * 1e3 / (2 * np.pi))
        ax.plot(tlist, Ht_list[2][1](tlist) * 1e3 / (2 * np.pi))
        ax.legend(("Real", "Imag"))
        ax.set_title("Qubit Drive")
        ax.set_xlabel("Time[ns]")
        ax.set_ylabel("$\Omega$[MHz]")

    return Ht_list