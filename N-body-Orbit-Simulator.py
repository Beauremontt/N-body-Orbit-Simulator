"""
Overview:
    2D N-body Orbit Simulator
    Uses Euler integration
    Class/objects:
        Vector, for position, velocity, and force
        Body, has mass, position, and velocity
        System, to track bodies and simulation steps
        All units are expected to be SI (m, kg, s)
"""

# %%

import math
import matplotlib.pyplot as plt


## Class initialization

class Vector:
    """Represents a 2D vector and has methods for basic vector operations."""
    
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y
    
    
    def __str__(self):
        """String customization for debugging."""
        return f"(x = {self.x:e}, y = {self.y:e})"
    
    # operation overloading for basic vector arithmetic (addition, scalar multiplication)
    # the static typing has quotes because the Vector class hasn't been defined yet
    def __add__(self, other: "Vector") -> "Vector":
        """Adds components."""
        new_x = self.x + other.x
        new_y = self.y + other.y
        return Vector(new_x, new_y)
    
    def __sub__(self, other: "Vector") -> "Vector":
        """Subtract components."""
        new_x = self.x - other.x
        new_y = self.y - other.y
        return Vector(new_x, new_y)
    
    def __mul__(self, scalar: float) -> "Vector":
        """Scalar multiplication: Vector * scalar."""
        if isinstance(scalar, (int, float)):
            new_x = self.x * scalar
            new_y = self.y * scalar
            return Vector(new_x, new_y)
        else:
            raise TypeError("Can only multiply Vector by an int or float.")
    
    def __rmul__(self, scalar: float) -> "Vector":  # __mul__ in reverse order
        """Scalar multiplication: scalar * Vector."""
        return self.__mul__(scalar)
    
    def __neg__(self) -> "Vector":
        """Unary negation: -Vector."""
        return Vector(-self.x, -self.y)
    
    def __truediv__(self, scalar: float) -> "Vector":
        """Scalar division: Vector / scalar."""
        if isinstance(scalar, (int, float)):
            if scalar == 0:
                raise ZeroDivisionError
            # multiply by reciprocal
            return self.__mul__(1 / scalar)
        else:
            raise TypeError("Can only divide Vector by an int or float.")
    
    
    def magnitude(self) -> float:
        """Method to get vector's length."""
        # Pythagoras theorem
        return math.sqrt(self.x**2 + self.y**2)
    
    def __abs__(self) -> float:
        """Overloads abs() to return the magnitude."""
        return self.magnitude()
    
    def normalize(self) -> "Vector":
        """Method to get vector's direction."""
        # calculate unit vector
        mag = abs(self)
        new_x = self.x / mag
        new_y = self.y / mag
        return Vector(new_x, new_y)
        


class Body:
    """Represents each mass in the system."""
    
    def __init__(self, name: str, mass: float, position: Vector, velocity: Vector):
        self.name = name
        self.mass = mass
        self.position = position
        self.velocity = velocity
        self.net_force = Vector(0, 0)
    
    
    def __str__(self):
        """String customization for debugging."""
        pos_x = self.position.x
        pos_y = self.position.y
        v_x = self.velocity.x
        v_y = self.velocity.y
        return f"'{self.name}': {self.mass} kg  @  ({pos_x:e}, {pos_y:e}) m  ->  ({v_x:e}, {v_y:e}) m/s"
    
    
    def compute_force(self, other: "Body") -> Vector:
        """Returns gravitational force exerted by another body."""
        
        GRAV_CONST = 6.6743e-11  # m3 kg-1 s-2 == N m2 kg-2
        r_displacement = self.position - other.position
        r_unit = r_displacement.normalize()
        
        # Newton's law of universal gravitation
        f_grav = -GRAV_CONST * (self.mass * other.mass) / abs(r_displacement)**2 * r_unit
        return f_grav



class System:
    """Manages the collection of bodies and advances the simulation."""
    
    def __init__(self, bodies: list[Body] = []):
        self.bodies = bodies

    
    def __iter__(self):
        """Allows iterating over bodies."""
        return iter(self.bodies)
    
    def __len__(self):
        return len(self.bodies)
    
    
    def add_body(self, body: Body):
        """Adds a body to the system."""
        self.bodies.append(body)
    
    
    def step(self, dt: float):
        """Steps the simulation by time dt."""
        
        # reset force accumulation
        for body in self:
            body.net_force = Vector(0, 0)
        
        # loop over all pairs to accumulate forces
        n = len(self)
        for i in range(n):
            for j in range(i + 1, n):
                body_1 = self.bodies[i]
                body_2 = self.bodies[j]
                
                # compute forces
                force_12 = body_1.compute_force(body_2)   # "force on 1 by 2"
                force_21 = -force_12  # Newton's 3rd law of motion
                
                # accumulate forces
                body_1.net_force += force_12
                body_2.net_force += force_21
        
        # update positions and velocities
        for body in self:
            # get new velocity via Euler method:
            # F = m a = m (v1 - v0) / dt => v1 = v0 + F / m * dt
            body.velocity += body.net_force / body.mass * dt
            
            # update position
            body.position += body.velocity * dt


## Function initialization

def adjust_central_body_velocity(bodies_data):
    """Adjusts the velocity of the heaviest body so total momentum is zero."""
    # assume the first body is the central one
    central_mass = bodies_data[0][1]
    
    # calculate total momentum
    total_px = 0
    total_py = 0
    for name, mass, x, y, vx, vy in bodies_data[1:]:
        total_px += mass * vx
        total_py += mass * vy
        
    # set central body's velocity to cancel total momentum
    central_vx = -total_px / central_mass
    central_vy = -total_py / central_mass
    
    # replace the central body's velocity
    name, mass, x, y, _, _ = bodies_data[0]
    bodies_data[0] = (name, mass, x, y, central_vx, central_vy)
    return bodies_data


def initialize_system(bodies_data: list) -> System:
    """Prepare hard-coded system for the Sun and its major planets (we live here!)."""
    # add each body to a system
    system = System()
    for name, mass, x, y, vx, vy in bodies_data:
        body = Body(name, mass, Vector(x, y), Vector(vx, vy))
        system.add_body(body)
    return system


def simulation(system: System, steps: int, dt: float) -> dict:
    """Drives the simulation, and returns a dictionary of trajectories."""
    
    # initialize a trajectory log
    trajectories = {body.name: [body.position] for body in system}
    
    # loop for each step
    for _ in range(steps):
        system.step(dt)
        
        # update trajectories
        for body in system:
            trajectories[body.name].append(body.position)
    
    return trajectories


## Hard-code choices of systems
solar_system_bodies_data = [
    # name,     mass,       x,          y,  vx, vy
    ('Sun',     1.9885e30,  0,          0,  0,  0),
    ('Mercury', 3.3011e23,  5.791e10,   0,  0,  -4.79e4),
    ('Venus',   4.8675e24,  1.082e11,   0,  0,  -3.5e4),
    ('Earth',   5.9722e24,  1.496e11,   0,  0,  -2.98e4),
    ('Mars',    6.4171e23,  2.279e11,   0,  0,  -2.41e4),
    ('Jupiter', 1.8982e27,  7.785e11,   0,  0,  -1.31e4),
    ('Saturn',  5.6834e26,  1.433e12,   0,  0,  -9.7e3),
    ('Uranus',  8.6810e25,  2.877e12,   0,  0,  -6.8e3),
    ('Neptune', 1.02413e26, 4.503e12,   0,  0,  -5.4e3),
]

earth_moon_bodies_data = [
    # name,     mass,           x,          y,  vx, vy
    ('Earth',   5.9722e24,      0,          0,  0,  0),
    ('Moon',    7.34767309e22,  3.844e8,    0,  0,  1022),
    ('ISS',     4.2e5,          6.771e6,    0,  0,  7670),
]

close_third_bodies_data = [  # Alpha-Beta double ring, with a very close fly-by
    # name,     mass (kg),   x (m),      y (m),      vx (m/s),   vy (m/s)
    ('Alpha',   2e30,        -1e11,      0,          0,          12000),
    ('Beta',    2e30,         1e11,      0,          0,         -12000),
    ('Interloper', 2e30,      0,         1.5e11,     -15000,     0),
]

far_third_bodies_data = [  # Alpha-Beta double ring, with a distant fly-by
    # name,     mass (kg),   x (m),      y (m),      vx (m/s),   vy (m/s)
    ('Alpha',   2e30,        -1e11,      0,          0,          12000),
    ('Beta',    2e30,         1e11,      0,          0,         -12000),
    # ('Interloper', 2e30,      1e12,         9e11,     -15000,     0),
]

stable_three_bodies_data = [  # "Lagrange Equilateral Triangel"
    # name,   mass (kg),   x (m),      y (m),      vx (m/s),    vy (m/s)
    ('A',     1e30,        1e11,       0,          0,           10000),
    ('B',     1e30,       -0.5e11,     0.866e11,  -8660,       -5000),
    ('C',     1e30,       -0.5e11,    -0.866e11,   8660,       -5000),
]

print("Choices of systems:")
print("  'Solar', 'Earth', 'Close Third', 'Far Third', 'Lagrange Triangle'")



# %%
## Main

# choose a bodies_data configuration
system_choice = input("Choose system: ")
match system_choice.lower():
    case 'solar':
        bodies_data = solar_system_bodies_data
    case 'earth':
        bodies_data = earth_moon_bodies_data
    case 'close third':
        bodies_data = close_third_bodies_data
    case 'far third':
        bodies_data = far_third_bodies_data
    case 'lagrange triangle':
        bodies_data = stable_three_bodies_data
    # case 'random':
    #     # TODO
    case _:
        print(f"\nUnknown choice '{system_choice}' selected.")
        exit()
    
        
# other chosen parameters
dt = float(input("Time between steps: "))
steps = int(input("Number of steps: "))

# adjust to get equilibrium (fixes wobbling of central body)
# only helps if the central body's mass >>> remaining mass
bodies_data = adjust_central_body_velocity(list(bodies_data))

# call the simulation
system = initialize_system(bodies_data)
trajectories = simulation(system, steps, dt)


# plot trajectories
plt.figure(figsize=(10, 10))

# pull each body's path
for name, positions in reversed(list(trajectories.items())):
    xs = [pos.x for pos in positions]
    ys = [pos.y for pos in positions]
    # draw on top of each other
    plt.plot(xs, ys)
    # add a marker at the last position
    plt.scatter(xs[-1], ys[-1], s=80, marker='o', label=f"{name}")

# add labels
plt.title('Orbital Simulation')
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.legend()
plt.axis('equal')  # circles will appear as circles
plt.grid(True)
plt.show()