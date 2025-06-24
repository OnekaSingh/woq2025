import numpy as np
import random

def anneal(x, z, explore, energy, cooling_rate):

    N = x.shape[0]

    x_opt, z_opt = x.copy(), z.copy()

    current_energy = energy(x, z)
    energy_opt = current_energy
    delta_energies = []

    for n in range(N):

        x, z = explore(n, x, z)

        new_energy = energy(x, z)

        delta_energies.append(np.abs(new_energy- current_energy)**2)

    energies = [current_energy]
    T = np.mean(delta_energies)

    T = N
    i = 0

    print(f"T0 = {T}")

    while T > 1-cooling_rate:

        n = random.randint(0, N-1)

        x, z = explore(n, x, z)
        new_energy = energy(x, z)

        if new_energy <= current_energy:

            current_energy = new_energy
            energies.append(current_energy)

            if current_energy < energy_opt:
                energy_opt = current_energy

                print(f"Iteration {i}", energy_opt)

                x_opt, z_opt = x.copy(), z.copy()

        else:
            p = np.exp(-(new_energy - current_energy)/T)
            #print(p)
            #print(new_energy - current_energy)
            # Accept worse solution with finite probability
            if random.random() < p:
                current_energy = new_energy
                energies.append(current_energy)

            # Undo operation
            else:
                x, z = explore(n, x, z)

        T *= cooling_rate
        i += 1

    return x_opt, z_opt, energies, energy_opt


def anneal(x, z, explore, energy, cooling_rate=0.999, min_temp=1e-3, max_iter=10000):
    N = x.shape[0]

    x_opt, z_opt = x.copy(), z.copy()
    current_energy = energy(x, z)
    energy_opt = current_energy

    # Estimate initial temperature using random perturbations
    delta_energies = []
    for n in range(N):
        x_trial, z_trial = explore(n, x.copy(), z.copy())
        delta_energies.append(np.abs(energy(x_trial, z_trial) - current_energy))

    T = np.mean(delta_energies) if delta_energies else 1.0  # Avoid division by zero

    energies = [current_energy]
    i = 0

    print(f"Initial temperature T0 = {T:.4f}")

    while T > min_temp and i < max_iter:
        n = random.randint(0, N - 1)

        # Save current state
        x_prev, z_prev = x.copy(), z.copy()

        # Explore new state
        x_new, z_new = explore(n, x.copy(), z.copy())
        new_energy = energy(x_new, z_new)

        delta_E = new_energy - current_energy

        if delta_E <= 0 or random.random() < np.exp(-delta_E / T):
            x, z = x_new, z_new
            current_energy = new_energy
            energies.append(current_energy)

            if current_energy < energy_opt:
                x_opt, z_opt = x.copy(), z.copy()
                energy_opt = current_energy
                print(f"Iteration {i}: New optimal energy = {energy_opt:.6f}")
        else:
            # Reject move, keep previous state
            x, z = x_prev, z_prev

        T *= cooling_rate
        i += 1

    return x_opt, z_opt, energies, energy_opt
