import matplotlib.pyplot as plt
import matplotlib
import random
import numpy as np
from matplotlib.animation import FuncAnimation


"""CONSTANTS"""
max_erbasts = 25
max_carvizes = 25
#days = 30
#NUMCELLS = 30
max_energy = 20
max_lifetime = 10


class Cell:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.herb = Herb()
        self.pride = Pride()
        self.veg = []
        self.temp_herb = []  #this is going to be a list of herbs
        self.temp_pride = []  # this is going to be a list of prides
        self.population_grow = []


    def find_near(self, world):
        all_near_cells = []
        earth_near_cells = []
        # i don't have the problem of going out of the grid because all the borders are water and i start only from the earth cells

        all_near_cells = [
        world[self.x + 1][self.y],   # Right
        world[self.x - 1][self.y],   # Left
        world[self.x][self.y - 1],   # Up (y-axis is reversed)
        world[self.x][self.y + 1]    # Down (y-axis is reversed)
        ]
        for cell in all_near_cells:
            if isinstance(cell, Earth):
                earth_near_cells.append(cell)
        return earth_near_cells

    def __repr__(self):
        return f'Cell at ({self.x},{self.y})'


class Water(Cell):

    def __init__(self, x, y):
        super().__init__(x, y)

    def population(self):
        return f'the cell is water'

    def __str__(self):
        return 'Water at ({self.x},{self.y})'


class Earth(Cell):

    def __init__(self, x, y):
        super().__init__(x, y)

    def add_inhabitant(self, inhabitant):
        if inhabitant == 'Erbast':
            new_inhabitant = Erbast()
            self.herb.add(new_inhabitant)

        elif inhabitant == 'Carviz':
            new_inhabitant = Carviz()
            self.pride.add(new_inhabitant)

    def add_vegetob(self):
        new_inhabitant = Vegetob()
        self.veg.append(new_inhabitant)

    def how_many_inhabitants(self):
        return self.pride.how_many() + self.herb.how_many()

    def population(self):
        return f'the cell has {self.how_many_inhabitants()} inhabitants: {self.pride.how_many()} Carviz, {self.herb.how_many()} Erbast and {self.density_of_vegetob()} vegetob'

    def density_of_vegetob(self):
        return f'{len(self.veg)} %'

    def __str__(self):
        return 'Earth at ({self.x},{self.y}'


class Inhabitant:

    def __init__(self):
        pass


class Vegetob(Inhabitant):

    def __init__(self):
        super().__init__()

    def __str__(self):
        return 'Vegetob'


class Erbast(Inhabitant):

    def __init__(self):
        super().__init__()
        global max_energy
        global max_lifetime
        self.energy = random.randint(1,max_energy)  # when it reaches 0, erbast dies
        self.lifetime = random.randint(1,max_lifetime)  # number of days erbast lives
        self.age = 0  # number of days from birth. when it reaches lifetime, erbast dies
        self.social = round(random.random(), 2)  # likelihood to join a herb (number between 0 and 1, rounded at 2 decimals)

    def __str__(self):
        return 'Erbast'


class Carviz(Inhabitant):

    def __init__(self):
        super().__init__()
        global max_energy
        global max_lifetime
        self.energy = random.randint(1,max_energy)  # when it reaches 0, carviz dies
        self.lifetime = random.randint(1,max_lifetime)  # number of days carviz lives
        self.age = 0  # number of days from birth. when it reaches lifetime, erbast dies
        self.social = round(random.random(), 2)  # likelihood to join a pride (number between 0 and 1, rounded at 2 decimals)

    def __str__(self):
        return 'Carviz'


class Group:
    def __init__(self):
        self.members = []


class Herb(Group):

    def __init__(self):
        super().__init__()

    def add(self, erbast):
        self.members.append(erbast)

    def remove(self, erbast):
        self.members.remove(erbast)

    def how_many(self):
        if self.members == []:
            return 0
        return len(self.members)


class Pride(Group):

    def __init__(self):
        super().__init__()

    def add(self, carviz):
        self.members.append(carviz)

    def remove(self, carviz):
        self.members.remove(carviz)

    def how_many(self):
        if self.members == []:
            return 0
        return len(self.members)


def create_grid(numcell):
    grid = []  # Initialize an empty grid
    global max_carvizes
    global max_erbasts

    for y in range(numcell):
        row = []
        for x in range(numcell):
            if x == 0 or x == numcell - 1 or y == 0 or y == numcell - 1:
                row.append(Water(y, x))  # Border cells are Water  # x y reversed
            else:
                cell_type = random.choice([Water, Earth])  # Randomly choose Water or Earth
                
                if cell_type == Earth:
                    cell = Earth(y, x)  # create the cell  # x y reversed

                    """choose the vegetobs"""
                    number_of_vegetob = random.randint(1,100)
                    for i in range(number_of_vegetob):
                        cell.add_vegetob()

                    """choose the erbasts"""
                    number_of_erbasts = random.randint(0, max_erbasts)
                    for i in range(number_of_erbasts):
                        cell.add_inhabitant('Erbast')

                    """choose the carvizes"""
                    number_of_carvizes = random.randint(0, max_carvizes)
                    for i in range(number_of_carvizes):
                        cell.add_inhabitant('Carviz')

                else:
                    cell = Water(y, x)
                row.append(cell)

        grid.append(row)

    return grid


def visualize_grid(grid):
    """returns a tuple of RGB colors for each cell"""

    def attribute_to_rgb(cell):

        if isinstance(cell, Water):
            return 0, 0, 0.3  # Blue for water
        
        else:
            vegetob_green = float(len(cell.veg) / 100)
            herb_blue = float(cell.herb.how_many() / max_erbasts)
            pride_red = float(cell.pride.how_many() / max_carvizes)
            return pride_red, vegetob_green, herb_blue  # RGB tuple

    # create a color array for each cell
    cell_colors = np.array([[attribute_to_rgb(cell) for cell in row] for row in grid])

    # create a custom colormap
    custom_cmap = matplotlib.colors.ListedColormap(cell_colors.reshape(-1, 3))

    # plot the grid with the custom colormap
    plt.imshow(cell_colors, cmap=custom_cmap, interpolation='nearest', aspect='equal')

    return cell_colors


def growing(grid):

    """increase the quantity of vegetobs"""
    for row in grid:
        for cell in row:
            if isinstance(cell, Earth):
                choose_increasing = random.randint(0, 40)  # vegetobs can grow at most 30%
                number_of_vegetobs = len(cell.veg)
                if choose_increasing != 0:
                    increment = (number_of_vegetobs * choose_increasing) // 100 + 1  # i add 1 in case the integer rounding gets to 0
                    for i in range(increment):
                        if len(cell.veg) < 100:
                            cell.add_vegetob()

                while len(cell.veg) < 10:  # i add this because i don't want the cell to be empty from the vegetobs
                    cell.add_vegetob()


    """check if a cell is completely surrounded by too much vegetob"""
    for row in grid:

        for cell in row:

            if isinstance(cell, Earth):

                if cell.find_near(grid) != []:

                    count = 0
                    
                    for near_cell in cell.find_near(grid):
                    
                        if len(near_cell.veg) == 100:
                            count += 1
                    
                    if count == len(cell.find_near(grid)):
                        cell.pride = Pride()
                        cell.herb = Herb()

    return grid


"""grazing is inside movement"""
def movement(grid):

    for row in grid:

        for cell in row:

            if isinstance(cell, Earth):

                remaining_herb = Herb()
                moving_herb = Herb()
                herb_choose = None
                remaining_pride = Pride()
                moving_pride = Pride()
                pride_choose = None
               


                """step 1"""
                """HERB MOVING"""
                now = 'herb is moving'  # i need this just for my understanding
                if now == 'herb is moving':

                    if cell.herb.members != []:

                        if cell.find_near(grid) == []:
                            herb_choose = 'stay'
                            for erbast in cell.herb.members:
                                remaining_herb.add(erbast)
                        
                        else:
                            
                            if len(cell.veg) < 10:
                                herb_choose = 'move'
                            else:
                                herb_choose = random.choice(['move', 'stay'])

                            if herb_choose == 'stay':
                                for erbast in cell.herb.members:
                                    remaining_herb.add(erbast)

                            else:  # if herb_choose == 'move'
                                new_cell_for_herb = random.choice(cell.find_near(grid))

                                for erbast in cell.herb.members:
                                    if erbast.energy < 5 or erbast.social < 0.3:
                                        remaining_herb.add(erbast)
                                    
                                    else:
                                        """choose the new cell"""
                                        
                                        if new_cell_for_herb.herb.how_many() + moving_herb.how_many() < max_erbasts:
                                            moving_herb.add(erbast)

                                        else:
                                            remaining_herb.add(erbast)

                """step 2"""
                """PRIDE MOVING"""
                now = 'pride is moving'  # i need this just for my understanding
                if now == 'pride is moving':

                    if cell.pride.members != []:

                        if cell.find_near(grid) == []:
                            pride_choose = 'stay'
                            for carviz in cell.pride.members:
                                remaining_pride.add(carviz)
                        
                        else:
                            
                            if len(cell.veg) < 10:
                                pride_choose = 'move'
                            else:
                                pride_choose = random.choice(['move', 'stay'])

                            if pride_choose == 'stay':
                                for carviz in cell.pride.members:
                                    remaining_pride.add(carviz)

                            else:  # if pride_choose == 'move'
                                new_cell_for_pride = random.choice(cell.find_near(grid))

                                for carviz in cell.pride.members:
                                    if carviz.energy < 5 or carviz.social < 0.3:
                                        remaining_pride.add(carviz)
                                    
                                    else:
                                        """choose the new cell"""
                                        
                                        if new_cell_for_pride.pride.how_many() + moving_pride.how_many() < max_carvizes:
                                            moving_pride.add(carviz)

                                        else:
                                            remaining_pride.add(carviz)


                """step 3"""
                """REMAINING_HERB are eating"""
                now = 'remaining herb'
                if now == 'remaining herb':
                    if remaining_herb.how_many() == 0:
                        pass
                    elif remaining_herb.how_many() >= len(cell.veg):  # eat based on their energy

                        """sort the individuals by their energy to decide who will eat"""
                        remaining_herb.members.sort(key=lambda erbast: erbast.energy)

                        initial_len_veg = len(cell.veg)
                        t = 0
                        for erbast in remaining_herb.members:
                            if t <= initial_len_veg:
                                """increase the energy and reduce vegetob"""
                                if erbast.energy < 20:
                                    erbast.energy += 1

                                cell.veg = cell.veg[:-1]
                                t += 1

                            else:
                                """reduce sociality"""
                                if erbast.social > 0:
                                    erbast.social -= 0.1

                                t += 1


                    else:  # remaining_herb.how_many() < len(cell.veg):

                        """everyone eats 1"""
                        for erbast in remaining_herb.members:
                            if erbast.energy < 20:
                                    erbast.energy += 1
                            cell.veg = cell.veg[:-1]


                """step 4"""
                """MOVING HERB are moving in the other cell"""
                now = 'moving herb'
                if now == 'moving herb':
                    if herb_choose == 'move':
                        new_cell_for_herb.temp_herb.append(moving_herb)


                """step 5"""
                """MOVING PRIDE is moving to the other cell"""
                now = 'moving pride'
                if now == 'moving pride':
                    if pride_choose == 'move':
                        new_cell_for_pride.temp_pride.append(moving_pride)

                """step 6"""
                """RIASSIGN HERB"""
                now = 'riassign herb'
                if now == 'riassign herb':
                    cell.herb = remaining_herb


                """step 7"""
                """RIASSIGN PRIDE"""
                now = 'riassign pride'
                if now == 'riassign pride':
                    cell.pride = remaining_pride

    return grid


def struggle(grid):
    for row in grid:
        for cell in row:
            if isinstance(cell, Earth):

                """step 1"""
                """about the herb: make everyone join"""
                now = 'step 1'
                if now == 'step 1':
                    if cell.temp_herb != []:
                        for herb in cell.temp_herb:
                            for erbast in herb.members:
                                if cell.herb.how_many() < max_erbasts:
                                    cell.herb.add(erbast)
                                    # all the erbasts that moved but didn't find a space to stay, they die
                                
                cell.temp_herb = []  # reset the temp_herb

                """step 2"""
                """about the pride"""
                join_list = []
                fight_list = []

                if cell.temp_pride != []:
                    for pride in cell.temp_pride:

                        if cell.pride.how_many() == 0:
                            fight_list.append(pride)

                        else:
                            
                            if cell.pride.how_many() + pride.how_many() + sum(pride.how_many() for pride in join_list) > max_carvizes:  # if there is not enough space for all the carvizes, they cannot join
                                choose = 'fight'

                            else:
                                
                                probability = sum(member.social for member in pride.members) / cell.pride.how_many()

                                if random.random() < probability:
                                    choose = 'join'
                                else:
                                    choose = 'fight'

                            if choose == 'join':
                                join_list.append(pride)
                            else:
                                fight_list.append(pride)

                """add the members that decided to join"""
                for pride in join_list:
                    for member in pride.members:
                        cell.pride.add(member)

                """start the fight"""
                """first, chose the strongest one in fight"""
                if fight_list != []:
                    while len(fight_list) != 1:
                        pride1 = random.choice(fight_list)
                        pride2 = random.choice(fight_list)
                        if pride1 != pride2:
                            winner = fight(pride1, pride2)
                            if winner == pride1:
                                fight_list.remove(pride2)
                            elif winner == pride2:
                                fight_list.remove(pride1)

                    """fight between the strongest opponent and the original one"""
                    if cell.pride.members == 0:
                        cell.pride = fight_list[0]
                    else:
                        new_pride = fight(cell.pride, fight_list[0])
                        cell.pride = new_pride

                    for carviz in cell.pride.members:
                        if carviz.social < 0.9:
                            carviz.social += 0.1
                            carviz.social = round(carviz.social, 2)
                        else:
                            carviz.social = 1

                cell.temp_pride = []  # reset the temp_pride

    return grid


def fight(pride1, pride2):
    force1 = sum(member.energy for member in pride1.members)
    force2 = sum(member.energy for member in pride2.members)
    if force1 > force2:
        return pride1
    elif force1 < force2:
        return pride2
    else:
        return random.choice([pride1, pride2])


def hunt(grid):
    for row in grid:
        for cell in row:
            if isinstance(cell, Earth):

                """find the strongest erbast"""
                if cell.herb.members != [] and cell.pride.members != []:
                    strongest_erbast = max(cell.herb.members, key=lambda erbast: erbast.energy)
                    strongest_carviz = max(cell.pride.members, key=lambda carviz: carviz.energy)

                    """fight"""
                    if strongest_carviz.energy >= strongest_erbast.energy:
                        """the erbast is defeated"""
                        cell.herb.remove(strongest_erbast)

                        """sort the pride"""
                        cell.pride.members.sort(key=lambda carviz: carviz.energy)

                        """increase the energy of the pride"""
                        divide_energy(strongest_erbast, cell.pride)


                    else:
                        """if no fight"""
                        for carviz in cell.pride.members:
                            carviz.social -= 0.1

                        if carviz.social < 0:  # to avoid complications
                            carviz.social = 0
    return grid

def divide_energy(strongest_erbast, pride):
    while strongest_erbast.energy >= pride.how_many():
        for member in pride.members:
            member.energy += 1
        strongest_erbast.energy -= pride.how_many()
        if strongest_erbast.energy < pride.how_many():
            break

    if strongest_erbast.energy > 0:
        t = 0
        for i in range(strongest_erbast.energy):
            pride.members[t].energy += 1
            t += 1


def spawning(grid):

    """increase age"""
    for row in grid:
        for cell in row:
            if isinstance(cell, Earth):

                while len(cell.veg) < 10:
                    cell.add_vegetob()

                temp1 = cell.herb.members

                for member in cell.herb.members:

                    member.age += 1

                    if member.age % 10 == 0:
                        member.energy -= 1
                
                for member in temp1:

                    if member.age == member.lifetime:
                        cell.herb.remove(member)

                        if cell.herb.how_many() < max_erbasts:

                            """generate first offspring"""

                            erbast1 = Erbast()
                            if member.energy > 0:
                                erbast1.energy = random.randint(0, member.energy)
                            else:
                                erbast1.energy = 0
                            if member.social > 0:
                                erbast1.social = round(random.uniform(0, member.social*2), 2)
                            else:
                                erbast1.social = 0
                            cell.herb.add(erbast1)


                            if cell.herb.how_many() < max_erbasts:

                                """generate second offspring"""

                                erbast2 = Erbast()
                                erbast2.energy = member.energy - erbast1.energy
                                erbast2.social = member.social*2 - erbast1.social
                                cell.herb.add(erbast2)

                for member in temp1:
                    if member.energy < 1:
                        cell.herb.remove(member)


                temp2 = cell.pride.members

                for member in cell.pride.members:

                    member.age += 1

                    if member.age % 10 == 0:
                        member.energy -= 1
                
                for member in temp2:

                    if member.age == member.lifetime:
                        cell.pride.remove(member)

                        if cell.pride.how_many() < max_carvizes:

                            """generate first offspring"""
                            carviz1 = Carviz()
                            if member.energy > 0:
                                carviz1.energy = random.randint(0, member.energy)
                            else:
                                carviz1.energy = 0
                            if member.social > 0:
                                carviz1.social = round(random.uniform(0, member.social*2), 2)
                            else:
                                carviz1.social = 0
                            cell.pride.add(carviz1)

                            if cell.pride.how_many() < max_carvizes:

                                """generate second offspring"""

                                carviz2 = Carviz()
                                carviz2.energy = member.energy - carviz1.energy
                                carviz2.social = member.social*2 - carviz1.social
                                cell.pride.add(carviz2)

                    for member in temp2:
                        if member.energy < 1:
                            cell.pride.remove(member)

    return grid


def pop_grow(grid):
    for row in grid:
        for cell in row:
            pop = collect_population_data(cell)
            cell.population_grow.append(pop)


def day(world):
    growing(world)
    movement(world)
    struggle(world)
    hunt(world)
    spawning(world)
    pop_grow(world)
    return world


"""
THIS IS JUST THE ANIMATION WITHOUT INTERACTIVE MODE


def update(frame):
  global world
  day(world)
  img.set_array(visualize_grid(world))
  return [img]


fig, ax = plt.subplots()
world = create_grid(NUMCELLS)


img = ax.imshow(visualize_grid(world), animated=True)

ani = FuncAnimation(fig, update, frames=days, interval=500, blit=True, repeat = False)

plt.show()  # display the animation"""



def collect_population_data(cell):
    veg = 0
    herb = 0
    pride = 0
    if isinstance(cell, Earth):
        veg = len(cell.veg)
        herb = cell.herb.how_many()
        pride = cell.pride.how_many()
    return {'veg': veg, 'herb': herb, 'pride': pride}

def plot_population_changes(population_data):
    days = range(len(population_data['veg']))

    plt.figure()
    plt.plot(days, population_data['veg'], label='Vegetobs')
    plt.plot(days, population_data['herb'], label='Erbasts')
    plt.plot(days, population_data['pride'], label='Carvizes')
    plt.xlabel('Days')
    plt.ylabel('Population')
    plt.title('Population Changes Over Time')
    plt.legend()
    plt.show()


def collect_tot_population(grid):
    veg_tot = 0
    herb_tot = 0
    pride_tot = 0
    for row in grid:
        for cell in row:
            if isinstance(cell, Earth):
                veg_tot += len(cell.veg)
                herb_tot += cell.herb.how_many()
                pride_tot += cell.pride.how_many()
    return {'veg': veg_tot, 'herb': herb_tot, 'pride': pride_tot}


def plot_tot_population(ax2, ax3, tot_population_history):
    days = range(len(tot_population_history['veg']))

    # vegetobs in the first subplot
    ax2.plot(days, tot_population_history['veg'], label='Vegetobs', color='green')
    ax2.set_xlabel('Days')
    ax2.set_ylabel('Population')
    ax2.set_title('Vegetation Changes')
    ax2.legend()
    ax2.grid(True)
    ax2.set_aspect('auto')

    # erbasts and carvizes in the second subplot
    ax3.plot(days, tot_population_history['herb'], label='Erbasts', color='blue')
    ax3.plot(days, tot_population_history['pride'], label='Carvizes', color='red')
    ax3.set_xlabel('Days')
    ax3.set_ylabel('Population')
    ax3.set_title('Population Changes')
    ax3.legend()
    ax3.grid(True)
    ax3.set_aspect('auto')


def collect_tot_energy(grid):
    herb_tot = 0
    pride_tot = 0
    herb_count = 0
    pride_count = 0
    for row in grid:
        for cell in row:
            if isinstance(cell, Earth):
                if cell.herb.how_many() != 0:
                    for erbast in cell.herb.members:
                        herb_tot += erbast.energy
                    herb_count += cell.herb.how_many()
                if cell.pride.how_many() != 0:
                    for carviz in cell.pride.members:
                        pride_tot += carviz.energy
                    pride_count += cell.pride.how_many()
    return {'herb': herb_tot / herb_count if herb_count else 0,  # to avoid division bi 0
            'pride': pride_tot / pride_count if pride_count else 0}



def plot_tot_energy(ax, tot_energy_history):
    days = range(len(tot_energy_history['herb']))

    # vegetobs in the first subplot
    ax.plot(days, tot_energy_history['herb'], label='Erbasts energy', color='blue')
    ax.plot(days, tot_energy_history['pride'], label='Carvizes energy', color='red')
    ax.set_xlabel('Days')
    ax.set_ylabel('Energy')
    ax.set_title('Energy Changes (total energy / # of members)')
    ax.legend()
    ax.grid(True)
    ax.set_aspect('auto')



def update(frame, img, world, ax2, ax3, ax4, tot_population_history, tot_energy_history, day_text):
    day(world)

    img.set_array(visualize_grid(world))

    population_now = collect_tot_population(world)    
    if len(tot_population_history['veg']) <= frame:
        tot_population_history['veg'].append(population_now['veg'])
        tot_population_history['herb'].append(population_now['herb'])
        tot_population_history['pride'].append(population_now['pride'])
    else:
        tot_population_history['veg'][frame] = population_now['veg']
        tot_population_history['herb'][frame] = population_now['herb']
        tot_population_history['pride'][frame] = population_now['pride']

    energy_now = collect_tot_energy(world)
    if len(tot_energy_history['herb']) <= frame:
        tot_energy_history['herb'].append(energy_now['herb'])
        tot_energy_history['pride'].append(energy_now['pride'])
    else:
        tot_energy_history['herb'][frame] = energy_now['herb']
        tot_energy_history['pride'][frame] = energy_now['pride']


    day_text.set_text(f"Day {frame + 1}")  # i will use this to display the day number

    # clear the plot to update the population
    ax2.clear()
    ax3.clear()
    plot_tot_population(ax2, ax3, tot_population_history)

    ax4.clear()
    plot_tot_energy(ax4, tot_energy_history)


    plt.draw()

    return [img]


instructions_text = (
    "To stop the animation, press the spacebar.\n"
    "To see the population of a cell, click on the bottom right part of the cell.\n"
    "Press the spacebar to start Planissus."
)



def run_animation():

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(18, 7))

    world = create_grid(NUMCELLS)

    img = ax1.imshow(visualize_grid(world), animated=True)
    
    
    tot_population_history = {'veg': [], 'herb': [], 'pride': []}  # prepare the population history
    tot_energy_history = {'herb': [], 'pride': []}

    day_text = ax1.text(0.5, 1.05, "", ha='center', va='center', transform=ax1.transAxes,
                        fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
    
    reminder1 = ax1.text(0.5, -0.2, "Press SPACE to pause or resume", ha='center', va='center', transform=ax1.transAxes,
                             fontsize=10, bbox=dict(facecolor='yellow', alpha=0.5))

    reminder2 = ax1.text(0.5, -0.3, "Click on the bottom right of a cell to visualize population", ha='center', va='center', transform=ax1.transAxes,
                                   fontsize=10, bbox=dict(facecolor='lightblue', alpha=0.5))


    ani = FuncAnimation(fig, update, fargs=(img, world, ax2, ax3, ax4, tot_population_history, tot_energy_history, day_text), frames=range(days), interval=500, blit=True, repeat=False)

    anim_running = True


    """PAUSE - PLAY"""
    def onKeyPress(event):
        nonlocal anim_running
        if event.key == ' ':
            if anim_running:
                ani.event_source.stop()
                anim_running = False
            else:
                ani.event_source.start()
                anim_running = True
    
    """visualize population in the cell by clicking it"""
    def onCellClick(event):
        if event.inaxes is not None:
            x = int(event.xdata)
            y = int(event.ydata)
            cell = world[y][x]

            data = cell.population_grow
            veg_values = [entry['veg'] for entry in data]
            erbast_values = [entry['herb'] for entry in data]
            carviz_values = [entry['pride'] for entry in data]
            indices = range(len(data))

            plt.figure(figsize=(10, 6))

            # i create a line for each inhabitant
            plt.plot(indices, veg_values, color='green', label='veg')
            plt.plot(indices, erbast_values, color='blue', linestyle='-', label='erbast')
            plt.plot(indices, carviz_values, color='red', linestyle='-', label='carviz')

            global days
            plt.ylim(-5, 105)
            plt.xlim(0, days)
            plt.title('Population changes')
            plt.legend()

            plt.grid(True)
            plt.tight_layout()
            plt.show()

    fig.canvas.mpl_connect('key_press_event', onKeyPress)
    fig.canvas.mpl_connect('button_press_event', onCellClick)

    plt.show()



"""INPUT FOR THE NUMCELLS AND DAYS"""
import tkinter as tk
from tkinter import simpledialog

def get_user_input():
    root = tk.Tk()
    root.withdraw()

    num_cells = simpledialog.askinteger("Input", "Enter the dimension of the grid:", minvalue=1, maxvalue=100)
    days = simpledialog.askinteger("Input", "Enter the number of days:", minvalue=1, maxvalue=1000)

    return num_cells, days




if __name__ == "__main__":
    NUMCELLS, days = get_user_input()
    run_animation()