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



