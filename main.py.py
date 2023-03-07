import sys
import functions

# Define variables for each command-line argument
timestep = 1
screen_x_min = 0
screen_y_min = 0
screen_x_max = 800
screen_y_max = 800
tile_size = 4
num_particles = 500
radius_max = 5
radius_min = 4
sim_time = 1500
plot_type = 0

# Parse the command-line arguments
for i in range(1, len(sys.argv), 2):
    flag = sys.argv[i]
    value = sys.argv[i+1]

    # Check which argument flag was provided, and set the corresponding variable
    if flag == "--timestep":
        timestep = float(value)
    elif flag == "--screen-x-min":
        screen_x_min = float(value)
    elif flag == "--screen-y-min":
        screen_y_min = float(value)
    elif flag == "--screen-x-max":
        screen_x_max = float(value)
    elif flag == "--screen-y-max":
        screen_y_max = float(value)
    elif flag == "--tile-size":
        tile_size = float(value)
    elif flag == "--num-particles":
        num_particles = int(value)
    elif flag == "--radius-max":
        radius_max = float(value)
    elif flag == "--radius-min":
        radius_min = float(value)
    elif flag == "--sim-time":
        sim_time = float(value)
    elif flag == "--plot-type":
        plot_type = int(value)

print("dt:", timestep )
print("screen-x-min:", screen_x_min )
print("screen-y-min:", screen_y_min )
print("screen-x-max:", screen_x_max )
print("screen-y-max:", screen_y_max)
print("tile-size:", tile_size )
print("num-particles:", num_particles)
print("radius-max:", radius_max )
print("radius-min:", radius_min )
print("sim-time:", sim_time  )
print("plot-type, 0=MSD, 1=Displacement:", plot_type)
# Call the simulation function with the parsed arguments
simulation = functions.Sim(
    timestep, 
    screen_x_min, 
    screen_y_min, 
    screen_x_max, 
    screen_y_max, 
    tile_size, 
    num_particles, 
    radius_max, 
    radius_min
)

simulation.run_with_plot(sim_time, plot_type)