# Pygame Collision Model for Scientific Computing Class

This project is a simple collision model built on pygame for the Scientific Computing class at Olin College of Engineering during the Spring semester of 2023. The motivation behind the model is the fact that elastic collision models can serve as simple models for colloidial glass transition. It uses uniform grid partitioning to speed up the search for colliding particles.

## Installation
To run the script, you will need to install Pygame. You can do this using pip:

```bash
pip install pygame
```

## Usage
To run the script, navigate to the project directory in your terminal and run the following command:


```bash
python main.py [OPTIONS]
```
Where OPTIONS are any of the following command-line arguments:

<ol>
     <li>--timestep: The size of each timestep in the simulation (default: 1) </li>
 <li>--screen-x-min: The minimum x coordinate of the Pygame window (default: 0)
 <li>--screen-y-min: The minimum y coordinate of the Pygame window (default: 0)
 <li>--screen-x-max: The maximum x coordinate of the Pygame window (default: 800)
 <li>--screen-y-max: The maximum y coordinate of the Pygame window (default: 800)
 <li>--tile-size: The size of each tile in the simulation (default: 4)
 <li>--num-particles: The number of particles in the simulation (default: 500)
 <li>--radius-max: The maximum radius of each particle (default: 5)
 <li>--radius-min: The minimum radius of each particle (default: 4)
 <li>--sim-time: The total time to run the simulation (default: 1500)
 <li>--plot-type: The type of plot to generate after the simulation is finished. 
 0 generates a Mean Square Distance vs Number of Particles plot, while 1 generates a Displacement vs Time plot (default: 0)
</ol>


For example, to run the simulation with a timestep of 0.5 and 1000 particles, use the following command:

```bash
python main.py --timestep 0.5 --num-particles 1000
```
## Results
Here are some sample graphs and gifs generated from the simulation:

![1200 Particles](/pics/1200.gif)


![8000 Particles](/pics/8000.gif)


Changing Slope of Mean Square Distance vs. Number of Particles

Mean Square Distance vs. Number of Particles

![Mean Square Distance vs. 50](/pics/msd50.PNG)

![Mean Square Distance vs. 1200](/pics/msd1200.PNG)

![Mean Square Distance vs. 3000](/pics/msd3000.PNG)


Brownian Motion of Particles with Increasing Number of Particles

![Brownian Motion 50 Particles](/pics/brownian_50.PNG)

Brownian Motion with 50 Particles

![Brownian Motion 1200 Particles](/pics/brownian_1200.png)

Brownian Motion with 1200 Particles

![Brownian Motion 3000 Particles](/pics/brownian_3000.PNG)

Brownian Motion with 3000 Particles

## Acknowledgements
This project was created as part of the Scientific Computing class at Olin College of Engineering in Spring 2023. The code was written by Samuel Cabrera Valencia. Special thanks to Dr. Carrie Nugent for guidance and support throughout the project.
