import random
import sys  # just for data update in terminal
import matplotlib.pyplot as plt
import numpy as np

# ────────────────────────────── Constants ────────────────────────────────
# Define simulation parameters and ecological constants

NUMCELLS = 100          # Size of the square world grid (100x100)
NUMDAYS = 500           # Number of simulation days
NEIGHBORHOOD = 1        # Radius of vision for movement decisions (Chebyshev distance)
WATER_RATE = 0.15       # Chance of a cell being water (non-inhabitable)

GROWING = 1             # Daily growth of vegetation (vegetob) per land cell

# Energy gain values
ENERGY_GAIN_GRAZE = 1   # Energy gained by erbasts from eating vegetob
ENERGY_GAIN_HUNT = 4    # Base energy gained by carvizes from a successful hunt

ENERGY_MOVE_COST = 1    # Energy cost for moving to a new cell

# Monthly aging penalties (every 10 days)
AGING_E = 1             # Energy loss for erbasts
AGING_C = 1             # Energy loss for carvizes

# Population caps (set to None for unlimited)
MAX_HERD = 1000           # Max erbasts per cell
MAX_PRIDE = 100          # Max carvizes per cell
MAX_ENERGY = 100         # Max energy an individual can store
MAX_LIFE = 100          # Max lifetime in days

# Daily chance of spontaneous birth per cell
ERBAST_BIRTH_RATE = 0.35
CARVIZ_BIRTH_RATE = 0.15

EPS = 1e-9              # Small epsilon to avoid division by zero (used in struggle and rgb)

# ────────────────────────────── Helpers ──────────────────────────────────

def cap(value, limit):
    """Clamp value to limit if limit is defined."""
    return min(value, limit) if limit is not None else value  # I use this to limit value to a maximum later in the code

def neighbours(i, j, radius, size):
    """
    Returns a list of valid neighbor coordinates within Chebyshev distance
    from a given grid position (i, j).
    """
    out = []
    for di in range(-radius, radius + 1):           # Loop over all possible offsets around the center point
        for dj in range(-radius, radius + 1):
            if di == 0 and dj == 0:                 # skip center cell
                continue
            ni, nj = i + di, j + dj                 # compute coordinates
            if 0 <= ni < size and 0 <= nj < size:   # Add neighbor if into the grid
                out.append((ni, nj))
    return out


# ───────────────────────────────────────── species ─────────────────────────────

class Erbast:
    """
    Erbast represents a herbivore. It eats vegetation to survive and can reproduce.
    """
    def __init__(self, energy=None, lifetime=None, age=0, social=None):
        self.energy = cap(energy if energy is not None else random.randint(4, 9), MAX_ENERGY)
        self.lifetime = cap(lifetime if lifetime is not None else random.randint(30, 90), MAX_LIFE)
        self.age = age
        self.social = social if social is not None else random.random()  # Likelihood to follow group movement
        self.has_moved = False                                           # Tracks whether this individual moved this turn

    @staticmethod                                                        # This because we do not use self in this method
    def newborn_from(parent):
        """Create a child from the parent's traits (same social and lifetime)."""
        return Erbast(energy=0, lifetime=parent.lifetime, age=0, social=parent.social)

    def apply_movement_cost(self):
        """Lose energy if moved."""
        if self.has_moved:
            self.energy -= ENERGY_MOVE_COST

    def monthly_aging(self):
        """Apply aging energy penalty every 10 days."""
        if self.age % 10 == 0:
            self.energy = max(0, self.energy - AGING_E)

    def alive(self):
        """Check if still alive based on energy and age."""
        return self.energy > 0 and self.age < self.lifetime


class Carviz:
    """
    Carviz represents a carnivore. It hunts erbasts and fights for dominance.
    """
    def __init__(self, energy=None, lifetime=None, age=0, social=None):
        self.energy = cap(energy if energy is not None else random.randint(6, 12), MAX_ENERGY)
        self.lifetime = cap(lifetime if lifetime is not None else random.randint(45, 110), MAX_LIFE)
        self.age = age
        self.social = social if social is not None else random.random()
        self.has_moved = False

    @staticmethod 
    def newborn_from(parent):
        """Create a child from the parent's traits."""
        return Carviz(energy=0, lifetime=parent.lifetime, age=0, social=parent.social)

    def apply_movement_cost(self):
        """Lose energy if moved."""
        if self.has_moved:
            self.energy -= ENERGY_MOVE_COST

    def monthly_aging(self):
        """Apply aging penalty every 10 days."""
        if self.age % 10 == 0:
            self.energy = max(0, self.energy - AGING_C)

    def alive(self):
        """Check for survival."""
        return self.energy > 0 and self.age < self.lifetime


# ───────────────────────────────────────── cell ────────────────────────────────

class Cell:
    __slots__ = ("is_water", "vegetob", "erbasts", "carvizes")      #this is just to make Cell object smaller in memory (it avoids creating dict for each instance)

    def __init__(self, is_water):
        self.is_water = is_water
        self.vegetob = 0 if is_water else random.randint(0, 100)    # Initial vegetation
        self.erbasts = []                                           # Herbivores in this cell
        self.carvizes = []                                          # Carnivores in this cell

    @property                                                       # This is used to call the method without parenthesis
    def herd_size(self):
        return len(self.erbasts)                                    # Number of erbasts

    @property
    def pride_size(self):
        return len(self.carvizes)                                   # Number of carvizes

    def rgb(self):
        """
        Converts the current state into an RGB color:
        - Red: carvizes (carnivores)
        - Green: erbasts (herbivores)
        - Blue: vegetob (vegetation)
        """
        r = min(1.0, self.pride_size / float(MAX_PRIDE/10 or self.pride_size + EPS))    # Makes the red color proprortional to the number of carnivores in the cell (we explicit cast to float to ensure precision)
        g = min(1.0, self.herd_size  / float(MAX_HERD/10  or self.herd_size  + EPS))    # Same for green (we divide by 10 just for a visual purpose, composite map is more readable)
        b = 0.0 if self.is_water else 0.4 - 0.3 * (self.vegetob / 100.0)                # 0 means water, 0.4 low vegetation, 0.1 max vegetation (we converto from a scale 0-100 to a 0.4-0.1 scale)
        return (r, g, b)


# ───────────────────────────────────────── world ───────────────────────────────

class World:
    def __init__(self, size):
        self.size = size
        self.day = 0
        self.grid = self._init_grid(size)

    # Initialize grid with cells
    def _init_grid(self, size):
        g = []
        for i in range(size):
            row = []
            for j in range(size):
                if i in (0, size - 1) or j in (0, size - 1):
                    row.append(Cell(True))  # Edges are always water
                else:
                    row.append(Cell(random.random() < WATER_RATE))
            g.append(row)

        # Populate with initial animals
        for i in range(size):
            for j in range(size):
                c = g[i][j]
                if not c.is_water:
                    if random.random() < 0.15:
                        c.erbasts.append(Erbast())
                    if random.random() < 0.05:
                        c.carvizes.append(Carviz())
        return g

    # Generator yielding all land (non-water) cells
    def _ground_cells(self):
        for row in self.grid:
            for c in row:
                if not c.is_water:
                    yield c              # Does not create a full list of cell in memory! 1 cell at time 


    # phase 1 – growing & overwhelm ------------------------------------
    def phase_growing(self):
        # Increase vegetob in all land cells
        for c in self._ground_cells():
            c.vegetob = min(100, c.vegetob + GROWING)  #add 2 vegetob to each land cells
        # Clear center cells completely surrounded by maxed-out vegetation
        to_clear = []  #store coords of cells that will be cleared for overgrowth
        for i in range(1, self.size - 1):
            for j in range(1, self.size - 1):
                cell = self.grid[i][j]
                if cell.is_water:
                    continue
                if all((not self.grid[ni][nj].is_water) and self.grid[ni][nj].vegetob == 100  #all is a function which checks if all the elements are true, here if all neighbors are land and fully overgrown
                       for ni, nj in neighbours(i, j, 1, self.size)):                         #we use neighbours() helper to get the cells in 1-cell chebyshev radius
                    to_clear.append((i, j))
        for i, j in to_clear:
            self.grid[i][j].erbasts.clear()
            self.grid[i][j].carvizes.clear()

    # phase 2 – movement ------------------------------------------------
    def phase_movement(self):
        size = self.size
        dest_map = [[(list(), list()) for _ in range(size)] for _ in range(size)] # Future cell state, each cell stores a tuple (list of erbast, list of carviz)
        for i in range(size):                                                     # Loop over every cell
            for j in range(size):
                cell = self.grid[i][j]
                if cell.is_water:
                    continue
                neigh = neighbours(i, j, NEIGHBORHOOD, size)
                ground_neigh = [(ni, nj) for ni, nj in neigh if not self.grid[ni][nj].is_water]     # Filter for only land
                 # Erbasts go where vegetob is highest
                target_herd = (i, j)
                if ground_neigh:
                    best_v = max(ground_neigh, key=lambda xy: self.grid[xy[0]][xy[1]].vegetob)      # Find cell with most vegetation
                    if self.grid[best_v[0]][best_v[1]].vegetob > cell.vegetob:
                        target_herd = best_v
                # Carvizes go where prey is most common
                target_pride = (i, j)
                if ground_neigh:
                    best_p = max(ground_neigh, key=lambda xy: self.grid[xy[0]][xy[1]].herd_size)   # Find cell with most prey 
                    if self.grid[best_p[0]][best_p[1]].herd_size > cell.herd_size:
                        target_pride = best_p
                # Move individuals based on social tendency
                for e in cell.erbasts:
                    if random.random() < e.social and target_herd != (i, j):                       # Check if its likelihood to follow herd
                        e.has_moved = True                                                         # Used later on for energy lost/grazing
                        dest_map[target_herd[0]][target_herd[1]][0].append(e)                      # Append to destination cell's erbast list
                    else:
                        e.has_moved = False
                        dest_map[i][j][0].append(e)
                for c_ in cell.carvizes:                                                           # Same logic as erbast
                    if random.random() < c_.social and target_pride != (i, j):
                        c_.has_moved = True
                        dest_map[target_pride[0]][target_pride[1]][1].append(c_)
                    else:
                        c_.has_moved = False
                        dest_map[i][j][1].append(c_)
        # Apply new positions
        for i in range(size):
            for j in range(size):
                cell = self.grid[i][j]
                cell.erbasts = dest_map[i][j][0]
                cell.carvizes = dest_map[i][j][1]

    # phase 3 – grazing -----------------------------------------------
    def phase_grazing(self):
        for cell in self._ground_cells():
            stayers = [e for e in cell.erbasts if not e.has_moved]  # Creates a list of erbasts which did not move
            if not stayers or cell.vegetob == 0:                    # No vegetation or no stayers = skip
                continue
            # Prioritize feeding those with least energy first
            stayers.sort(key=lambda e: e.energy)                    # Sort the stayers (those with less energy to the front)
            edible = min(cell.vegetob, len(stayers))
            for idx in range(edible):
                eater = stayers[idx]
                eater.energy = cap(eater.energy + ENERGY_GAIN_GRAZE, MAX_ENERGY)
            # Reduce vegetation based on grazers
            cell.vegetob -= edible
            # Slightly reduce social behavior of unfed erbasts
            for unfed in stayers[edible:]:  # little social penalty
                unfed.social = max(0.0, unfed.social - 0.05)

    # phase 4 – struggle (fight & hunt)--------------------------------
    def phase_struggle(self):
        for cell in self._ground_cells():
            # Internal pride fights until only one carviz remains
            while cell.pride_size > 1:                                          # We need cells with more than 1 Carviz
                a = cell.carvizes.pop()
                b = cell.carvizes.pop()
                total = a.energy + b.energy + EPS
                if random.random() < a.energy / total:                          # Winner choosen probabilistically based on energy
                    a.energy = cap(a.energy + b.energy // 2, MAX_ENERGY)
                    cell.carvizes.append(a)
                else:
                    b.energy = cap(b.energy + a.energy // 2, MAX_ENERGY)
                    cell.carvizes.append(b)
            # Hunting: single surviving carviz may try to kill erbast
            if cell.pride_size > 0 and cell.herd_size > 0:                      # If there at least 1 Erbast and 1 Carviz in the cell
                prey = max(cell.erbasts, key=lambda e: e.energy)                # Carviz chooses the strongest erbast
                pride_energy = cell.carvizes[0].energy  # One remaining hunter
                p_success = pride_energy / (pride_energy + prey.energy + EPS)   # Probability of success (Higher if predator stronger than prey)
                if random.random() < p_success:
                     # Success: erbast is killed, energy distributed to the carviz(es)
                    cell.erbasts.remove(prey)
                    gain = ENERGY_GAIN_HUNT + prey.energy
                    share = gain // cell.pride_size                             # Energy evenly split
                    remainder = gain - share * cell.pride_size
                    for c_ in cell.carvizes:
                        c_.energy = cap(c_.energy + share, MAX_ENERGY)
                    if remainder:
                        weakest = min(cell.carvizes, key=lambda c_: c_.energy)
                        weakest.energy = cap(weakest.energy + remainder, MAX_ENERGY)    # Remainder to the weakest
                else:
                    # Failed hunt: carvizes lose energy and social trust
                    for c_ in cell.carvizes:
                        c_.energy = max(0, c_.energy - 1)
                        c_.social = max(0.0, c_.social - 0.03)

    # phase 5 – spawning & ageing -------------------------------------
    def phase_spawning(self):
        for cell in self._ground_cells():
            new_erb = []                        # New lists to build next gen
            new_car = []
            # Update erbasts
            for e in cell.erbasts:
                e.apply_movement_cost()
                e.age += 1
                e.monthly_aging()
                if e.age >= e.lifetime:
                    # If dying, try to split into two offspring if space allows
                    if (MAX_HERD is None or cell.herd_size + len(new_erb) + 2 <= MAX_HERD):         # Check if there is space in the cell
                        ch1 = Erbast.newborn_from(e)
                        ch2 = Erbast.newborn_from(e)
                        share = e.energy // 2
                        ch1.energy = share
                        ch2.energy = e.energy - share
                        new_erb.extend([ch1, ch2])       # Extend is used to add the elements of the list to the list instead of all the object as a single element
                    # parent dies regardless
                elif e.alive():
                    new_erb.append(e)
            cell.erbasts = new_erb
            # Update carvizes (same as erbasts)
            for c in cell.carvizes:
                c.apply_movement_cost()
                c.age += 1
                c.monthly_aging()
                if c.age >= c.lifetime:
                    if (MAX_PRIDE is None or cell.pride_size + len(new_car) + 2 <= MAX_PRIDE):
                        ch1 = Carviz.newborn_from(c)
                        ch2 = Carviz.newborn_from(c)
                        share = c.energy // 2
                        ch1.energy = share
                        ch2.energy = c.energy - share
                        new_car.extend([ch1, ch2])
                elif c.alive():
                    new_car.append(c)
            cell.carvizes = new_car
            # Random spontaneous births (even if no parents are present)
            if (MAX_HERD is None or cell.herd_size < MAX_HERD) and random.random() < ERBAST_BIRTH_RATE:
                cell.erbasts.append(Erbast())
            if (MAX_PRIDE is None or cell.pride_size < MAX_PRIDE) and random.random() < CARVIZ_BIRTH_RATE:
                cell.carvizes.append(Carviz())

    # day wrapper ------------------------------------------------------
    def simulate_day(self):
        # Run all 5 ecological phases
        self.phase_growing()
        self.phase_movement()
        self.phase_grazing()
        self.phase_struggle()
        self.phase_spawning()
        self.day += 1

    # data aggregation for plots --------------------------------------
    def totals(self):
        veg = erb = car = 0
        for c in self._ground_cells():
            veg += c.vegetob
            erb += c.herd_size
            car += c.pride_size
        return veg, erb, car

    def maps(self):
        # Generate matrices for visualization
        veget = np.zeros((self.size, self.size), dtype=int)
        erbm  = np.zeros_like(veget)
        carm  = np.zeros_like(veget)
        rgb   = np.zeros((self.size, self.size, 3), dtype=float)  # Extra dimension for the 3 color channels
        for i in range(self.size):                                # We traverse all the grid cell by cell and get the data
            for j in range(self.size):
                c = self.grid[i][j]
                veget[i, j] = c.vegetob
                erbm[i, j] = c.herd_size
                carm[i, j] = c.pride_size
                rgb[i, j] = c.rgb()
        return veget, erbm, carm, rgb

# ───────────────────────────────────────── main loop ─────────────────────────--

def main():
    world = World(NUMCELLS)
    veg_tr, erb_tr, car_tr = [], [], []

    plt.ion()                               # Figure updates without blocking the code
    fig = plt.figure(figsize=(14, 12))
    plt.show(block=False)

    for _ in range(NUMDAYS):
        world.simulate_day()
        
        # Collect totals
        v, e, c = world.totals()
        veg_tr.append(v)
        erb_tr.append(e)
        car_tr.append(c)
        
        # Console status
        sys.stdout.write(f"\rDay {world.day:3d}  Vegetob={v:6d}  Erbast={e:4d}  Carviz={c:4d}")     # 3d,6d,4d refer to the number of digit alignment
        sys.stdout.flush()

        # Generate map layers
        vmap, emap, cmap, rgb = world.maps()

        # Clear and redraw all plots
        plt.clf()
        
        # composite RGB map
        ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=3)
        ax1.imshow(rgb, interpolation="nearest")            #interpolation used not to have smooth transition between pixels
        ax1.set_title(f"Composite map – day {world.day}")
        ax1.axis("off")

        # Vegetob heatmap
        ax2 = plt.subplot2grid((3, 3), (1, 0))
        im2 = ax2.imshow(vmap, cmap="YlGn", interpolation="nearest")
        ax2.set_title("Vegetob density")
        plt.colorbar(im2, ax=ax2, fraction=0.046)           # Fraction controls the width of the colorbar

        # Erbast heatmap
        ax3 = plt.subplot2grid((3, 3), (1, 1))
        im3 = ax3.imshow(emap, cmap="Blues", interpolation="nearest")
        ax3.set_title("Erbast population")
        plt.colorbar(im3, ax=ax3, fraction=0.046)

        # Carviz heatmap
        ax4 = plt.subplot2grid((3, 3), (1, 2))
        im4 = ax4.imshow(cmap, cmap="Reds", interpolation="nearest")
        ax4.set_title("Carviz population")
        plt.colorbar(im4, ax=ax4, fraction=0.046)

        # Trends plot
        ax5 = plt.subplot2grid((3, 3), (2, 0), colspan=3)
        d = np.arange(1, world.day + 1)
        ax5.plot(d, veg_tr, "g-", label="Vegetob")
        ax5.plot(d, erb_tr, "b-", label="Erbast")
        ax5.plot(d, car_tr, "r-", label="Carviz")
        ax5.set_yscale("log")
        ax5.set_xlabel("Day")
        ax5.set_ylabel("Total (log scale)")
        ax5.set_title("Population & resource trend")
        ax5.grid(True)
        ax5.legend()

        plt.tight_layout()              # Adjusts spacing between subplots
        plt.pause(0.25)

    plt.ioff()                          # Turn off interactive plotting
    plt.show()                          # Keeps the final figure open

# Run the simulation
if __name__ == "__main__":
    main()
