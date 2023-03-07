import pygame, math, sys
import numpy as np
import itertools
import matplotlib.pyplot as plt
import threading

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 50, 50)
YELLOW = (255, 255, 0)
GREEN = (0, 255, 50)
BLUE = (50, 50, 255)
CYAN = (0, 255, 255)
MAGENTA = (255, 0, 255)
BACKGROUND = (34, 40, 44)


class Box():
    #Class for defining methods to do with UGP
    def __init__(self, x_0, y_0, x_max, y_max,size, surface):
        self.x_max = x_max
        self.y_max = y_max
        self.screen_width = x_max
        self.screen_height =y_max
        self.tile_size= size
        self.x_0 = x_0
        self.y_0 = y_0
        self.tiles, self.num_tiles_x, self.num_tiles_y = self.define_grid(size)
    def define_grid(self, size):
    # Calculate the number of tiles in x and y direction
        num_tiles_x = int(math.ceil(self.screen_width / self.tile_size))
        num_tiles_y = int(math.ceil(self.screen_height / self.tile_size))

        # Create the tiles
        tiles = []
        for i in range(num_tiles_x):
            for j in range(num_tiles_y):
                # Define the tile as a pygame.Rect object
                tile_x = i * self.tile_size
                tile_y = j * self.tile_size
                tile = pygame.Rect(tile_x, tile_y, self.tile_size, self.tile_size)

                # Add the tile to the list of tiles
                tiles.append(tile)

        return tiles, num_tiles_x, num_tiles_y
    def get_colliding_tiles(self, part_l):
    # Create a set to store tiles that contain more than one circle
        colliding_tiles = set()

        # Create a dictionary to store the circle indices for each tile
        tile_circles = {}

        # Loop through all circles and add them to the dictionary for each tile they overlap with
        for i, part in enumerate(part_l):
            # Determine the bounding box of the circle
            x, y, r = part.pos[0], part.pos[1], part.R
            bbox = pygame.Rect(x - r, y - r, 2 * r, 2 * r)

            # Determine the range of tiles that the bounding box of the circle overlaps with
            min_x = int(bbox.left // self.tile_size)
            max_x = int(bbox.right // self.tile_size) + 1
            min_y = int(bbox.top // self.tile_size)
            max_y = int(bbox.bottom // self.tile_size) + 1

            # Add the circle to the dictionary for each tile it overlaps with
            for x, y in itertools.product(range(min_x, max_x), range(min_y, max_y)):
                if x >= 0 and x < self.num_tiles_x and y >= 0 and y < self.num_tiles_y:
                    tile_index = y * self.num_tiles_x + x
                    if tile_index not in tile_circles:
                        tile_circles[tile_index] = []
                    tile_circles[tile_index].append(i)
                    
        return colliding_tiles, tile_circles

class Particle():
    def __init__(self, posx, posy, R,tag,surface):
        self.pos = (posx, posy)
        self.vx = np.random.normal(0, 1)
        self.vy = np.random.normal(0, 1)
        #self.vx = np.random.randint()
        self.v = (self.vy, self.vx)
        self.R = R 
        self.m = np.random.randint(1,4)
        self.tag = tag
      
        self.col = (abs(int(self.vx*50)), abs(int(self.vy*50)), self.m*4)
        #self.col = (self.m*63.75,50,50)
        self.circ = pygame.draw.circle(surface, self.col, (posx,posy), R)
    def set_new_vars(self, posx_array, posy_array, velx_array, vely_array,surface):
        self.pos = (posx_array[self.tag], posy_array[self.tag])
        self.vx = velx_array[self.tag]
        self.vy = vely_array[self.tag]
        self.v = (self.vx, self.vy)
        #self.circ = pygame.draw.circle(surface, self.col, self.pos, self.R)

    def set_vel(self, vx, vy):
        self.vx = vx
        self.vy = vy
        self.v = (self.vx, self.vy)
    def step(self ,x_0, y_0, x_max, y_max,surface):
        
        new_pos_x = self.pos[0] + self.vx
        new_pos_y = self.pos[1] + self.vy       
        if new_pos_x+self.R > x_max:
            self.vx *= -1
            new_pos_x = x_max - (self.R)
        if new_pos_x-self.R < x_0:
            self.vx *= -1
            new_pos_x = x_0 + self.R
        if new_pos_y+self.R > y_max:
            self.vy *= -1
            new_pos_y = y_max - self.R
        if new_pos_y-self.R < y_0:
            self.vy *= -1
            new_pos_y = y_0 + self.R

        self.pos = (new_pos_x, new_pos_y)
        
        self.circ = pygame.draw.circle(surface, self.col, (new_pos_x, new_pos_y), self.R)
    def collides_with(self, other):
        distance = math.sqrt((self.pos[0] - other.pos[0])**2 + (self.pos[1] - other.pos[1])**2)
        return distance <= self.R + other.R
    
class Sim():
    def __init__(self,dt=1,x_0=0, y_0=0, x_max=800, y_max=800,size=5, num_particles = 500, R_max= 5, r_min=4):
        pygame.init()
        self.dt = dt
        X = x_max  # screen width
        Y = y_max  # screen 
        self.x_0 = x_0
        self.y_0 = y_0
        self.x_max = x_max
        self.y_max = y_max
        self.surface = pygame.display.set_mode((X, Y))
        self.surface.fill(WHITE)
        self.T_l = ((x_0, y_0), (x_max, y_0))
        self.R_l = ((x_max, y_0), (x_max, y_max))
        self.L_l = ((x_0, y_0), (x_0, y_max))
        self.B_l = ((x_0, y_max), (x_max, y_max))
        self.num_particles = num_particles
        self.Radius_max = R_max
        self.box = Box(x_0,y_0, x_max, y_max,size,self.surface)
        self.part_l = []
        radii = []
        self.posx_list = []
        self.posy_list = []
        self.velx_list = []

        self.vely_list = []
        for i in range(num_particles):
          
            r = np.random.randint(r_min, self.Radius_max)
            if any(abs(r - x) < x_max//num_particles for x in radii):
                r = np.random.randint(r_min, self.Radius_max)
            radii.append(r)
            
            x = np.random.randint(x_0+r*2, x_max-r*2)
            y = np.random.randint(y_0+r*2, y_max-r*2)
            self.part_l.append(Particle(x, y, r,i, self.surface))
            self.posx_list.append(x)
            self.posy_list.append(y)
            self.velx_list.append(self.part_l[i].vx)
            self.vely_list.append(self.part_l[i].vy)
    
        self.posx_array = np.array(self.posx_list)
        self.posy_array = np.array(self.posy_list)
        self.velx_array = np.array(self.velx_list)
        self.vely_array = np.array(self.vely_list)
        self.radii_array = np.array(radii)
        self.pos_array = np.array(list(zip(self.posx_array, self.posy_array)))
        self.vel_array = np.array(list(zip(self.velx_array, self.vely_array)))
        self.pos_arrays = [self.pos_array]
        self.vel_arrays = [self.vel_array]
        self.clock =pygame.time.Clock()

    def step(self):
        fake_screen = self.surface.copy()
        #fake_screen.fill(WHITE)
        #pygame.draw.line(fake_screen, BLUE, self.T_l[0], self.T_l[1], width = 10) 
        #pygame.draw.line(fake_screen, BLUE, self.R_l[0], self.R_l[1], width = 10) 
        #pygame.draw.line(fake_screen, BLUE, self.L_l[0], self.L_l[1], width = 10) 
        #pygame.draw.line(fake_screen, BLUE, self.B_l[0], self.B_l[1], width = 10) 
        

        test_1, test_2 = self.box.get_colliding_tiles(self.part_l)
        colls_to_c = self.check_colls(test_2)
        if colls_to_c != set():
            for i in colls_to_c:
               
                if self.part_l[i[0]].collides_with(self.part_l[i[1]]):
        
                    vx1f, vy1f, vx2f, vy2f, = self.collision(self.part_l[i[0]], self.part_l[i[1]], fake_screen)
                
                    self.velx_array[self.part_l[i[0]].tag] = vx1f
                    self.velx_array[self.part_l[i[1]].tag] = vx2f
                    self.vely_array[self.part_l[i[0]].tag] = vy1f
                    self.vely_array[self.part_l[i[1]].tag]= vy2f
                    self.posx_array[self.part_l[i[0]].tag] = self.part_l[i[0]].pos[0] + vx1f
                    self.posx_array[self.part_l[i[1]].tag]= self.part_l[i[1]].pos[0] + vx2f
                    self.posy_array[self.part_l[i[0]].tag]= self.part_l[i[0]].pos[1] + vy1f
                    self.posy_array[self.part_l[i[1]].tag] = self.part_l[i[1]].pos[1] + vy2f
                  
        new_posx_array = np.add(self.posx_array, self.velx_array*self.dt)
        new_posy_array = np.add(self.posy_array, self.vely_array*self.dt)

 
        new_posx_array[(new_posx_array + self.radii_array) >= self.x_max] = self.x_max-self.radii_array[(new_posx_array + self.radii_array) >= self.x_max]
        #print(new_posx_array)

        new_posx_array[(new_posx_array - self.radii_array) <= self.x_0] = self.x_0+self.radii_array[(new_posx_array - self.radii_array) <= self.x_0]
        #print(new_posx_array)
        #break
        new_posy_array[(new_posy_array + self.radii_array) >= self.y_max] = self.y_max-self.radii_array[(new_posy_array + self.radii_array) >= self.y_max]
        new_posy_array[(new_posy_array - self.radii_array) <= self.y_0] = self.y_0+self.radii_array[(new_posy_array - self.radii_array) <= self.y_0]
        new_velx_array = np.copy(self.velx_array)
        new_vely_array = np.copy(self.vely_array)
        #print(new_velx_array)
        new_velx_array[(new_posx_array + self.radii_array) >= self.x_max] *= -1
        new_velx_array[(new_posx_array - self.radii_array) <= self.x_0] *= -1
        new_vely_array[(new_posy_array + self.radii_array) >= self.y_max] *= -1 
        new_vely_array[(new_posy_array - self.radii_array) <= self.y_0] *=  -1
        self.posx_array = new_posx_array
        self.posy_array = new_posy_array
        self.velx_array = new_velx_array
        self.vely_array = new_vely_array
        for part in self.part_l:
            part.set_new_vars(self.posx_array, self.posy_array, self.velx_array, self.vely_array,fake_screen)

        self.pos_array = np.array(list(zip(self.posx_array, self.posy_array)))
        self.vel_array = np.array(list(zip(self.velx_array, self.vely_array)))
        self.pos_arrays.append(self.pos_array)
        self.vel_arrays.append(self.vel_array)

        #self.surface.blit(fake_screen, (0, 0))
        
        #self.seed.step(self.surface)
        #pygame.display.flip()
        return self.pos_array, self.vel_array
    def check_colls(self, tile_circles):
        colls_to_check = set()
        if tile_circles != {}:
            for check_list in tile_circles.values():
                if len(check_list)==2:
                    colls_to_check.add(tuple(check_list))
                else:
                    combs = itertools.combinations(check_list, 2)
                    for i in combs:
                        colls_to_check.add(i)
        return colls_to_check
    def collision(self,part1, part2, fake_screen):
    # Calculate the initial velocities' components
        m1 = part1.m
        m2 = part2.m
        x1i = part1.pos[0]
        y1i = part1.pos[1]
        x2i = part2.pos[0]
        y2i = part2.pos[1]
        v1ix = part1.vx
        v1iy = part1.vy
        v2ix = part2.vx
        v2iy = part2.vy
        
        # Calculate the final velocities' components
        vx1f =  ((m1 - m2) * v1ix + 2 * m2 * v2ix) / (m1 + m2)
        vy1f =  ((m1 - m2) * v1iy + 2 * m2 * v2iy) / (m1 + m2)
        vx2f = ((m2 - m1) * v2ix + 2 * m1 * v1ix) / (m1 + m2)

        vy2f = ((m2 - m1) * v2iy + 2 * m1 * v1iy) / (m1 + m2)
    
        
         # Normalize the final velocities to ensure that the maximum velocity is 1.0
        max_vel = max(abs(vx1f), abs(vy1f), abs(vx2f), abs(vy2f))
        if max_vel > 0.0:
            vx1f /= max_vel
            vy1f /= max_vel
            vx2f /= max_vel
            vy2f /= max_vel
        
        # Calculate the final velocities' magnitudes and directions
      
     
        new_pos1x = part1.pos[0] + vx1f*self.dt
        new_pos1y = part1.pos[1] + vy1f*self.dt
        new_pos2x = part2.pos[0] + vx2f*self.dt
        new_pos2y = part2.pos[1] + vy2f*self.dt
        # Check if the particles are still overlapping and move them apart if necessary
        dist = math.sqrt((new_pos1x - new_pos2x) ** 2 + (new_pos1y - new_pos2y) ** 2)
        overlap = (part1.R + part2.R) - dist
        pos1_list = list(part1.pos)
        pos2_list = list(part2.pos)
        
        if overlap > 0:
            nudge = overlap/2
            
            # Move the particles apart in proportion to their masses
            total_mass = m1 + m2
            new_pos1x += nudge * (m2 / total_mass) * (new_pos1x - new_pos2x) / dist
            new_pos1y += nudge * (m2 / total_mass) * (new_pos1y - new_pos2y) / dist
            new_pos2x -= nudge * (m1 / total_mass) * (new_pos1x - new_pos2x) / dist
            new_pos2y -= nudge * (m1 / total_mass) * (new_pos1y - new_pos2y) / dist
            
            part1.pos = (new_pos1x, new_pos1y)
            part2.pos = (new_pos2x, new_pos2y)
            
            

        return vx1f, vy1f, vx2f, vy2f
    def draw_balls(self):
        fake_screen = self.surface.copy()
        fake_screen.fill(WHITE)
        pygame.draw.line(fake_screen, BLUE, self.T_l[0], self.T_l[1], width = 10) 
        pygame.draw.line(fake_screen, BLUE, self.R_l[0], self.R_l[1], width = 10) 
        pygame.draw.line(fake_screen, BLUE, self.L_l[0], self.L_l[1], width = 10) 
        pygame.draw.line(fake_screen, BLUE, self.B_l[0], self.B_l[1], width = 10) 
        for i, pos in enumerate(self.pos_array):
            self.circ = pygame.draw.circle(fake_screen, self.part_l[i].col, pos, self.part_l[i].R)
        self.surface.blit(fake_screen, (0, 0))
        pygame.display.flip()
        self.clock.tick(60)

    def draw_all_balls(self):
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
            for j in self.pos_arrays:
                fake_screen = self.surface.copy()
                fake_screen.fill(WHITE)
                pygame.draw.line(fake_screen, BLUE, self.T_l[0], self.T_l[1], width = 10) 
                pygame.draw.line(fake_screen, BLUE, self.R_l[0], self.R_l[1], width = 10) 
                pygame.draw.line(fake_screen, BLUE, self.L_l[0], self.L_l[1], width = 10) 
                pygame.draw.line(fake_screen, BLUE, self.B_l[0], self.B_l[1], width = 10) 
                for i, pos in enumerate(j):
                    self.circ = pygame.draw.circle(fake_screen, self.part_l[i].col, pos, self.part_l[i].R)
                self.surface.blit(fake_screen, (0, 0))
                pygame.display.flip()
                self.clock.tick(60)
            break

    def run_sim(self, time=1500):
        for i in range(time):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
            self.step()
        self.draw_all_balls()
    def run_sim_timed(self, time=1500, plot = None):
      
        for i in range(time):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
            self.step()
            #if i % 2 == 0:
            self.draw_balls()
        #self.plot_mean_displacement()
    def run_with_plot(self, time= 1500, plot = 0):
        #Run Sim for set time, plot
        pygame_thread = threading.Thread(target=self.run_sim_timed)
        pygame_thread.start()

        # Wait for the Pygame thread to exit
        pygame_thread.join()
        if plot == 0:
            plot_thread = threading.Thread(target=self.plot_mean_square_distance)
            plot_thread.start()
       
        plot_thread_2 = threading.Thread(target=self.plot_pos)
        plot_thread_2.start()
    def plot_pos(self):
        x_vals = []
        y_vals = []
        for i in range(len(self.pos_arrays)):
            x_vals.append(self.pos_arrays[i][0][0])
            y_vals.append(self.pos_arrays[i][0][1])
     
        #fig, ax = plt.subplots()
        #print(x_vals)
        # Plot the x and y coordinate lists on the axis
        plt.plot(x_vals, y_vals)
        plt.title(f"Particle Trajectory Over Time for {self.num_particles} Particles")
        plt.xlabel('X Displacement')
        plt.ylabel('Y Displacement')
        plt.show()
        # Add labels and a title to the plot
        #plt.set_xlabel('X')
        #plt.set_ylabel('Y')
        #plt.set_title('Particle Trajectory')

    def run_sim_live(self):
       
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
            self.step()
            self.draw_balls()
    
    def plot_mean_square_distance(self, axis=0):
        msd = [0]
        for i in range(len(self.pos_arrays)-1):
            pos1 = np.array(self.pos_arrays[i])
           
            pos2 = np.array(self.pos_arrays[0])
            displacement = (pos2- pos1)
            squared_displacement = displacement**2
       
            msd.append(np.mean(squared_displacement[:,axis]))
        time_steps = [i for i in range(len(msd))]
      
        slope=  np.diff(np.log(msd)) / np.diff(np.log(time_steps))
        slope = slope[20:]
        print(slope)
        plt.scatter(range(len(msd)), msd)
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Time')
        
        plt.ylabel('Mean Square Displacement')
        plt.title(f"Mean Square Displacement over Time for {self.num_particles} Particles, slope = {np.mean(slope)}")

        plt.show()

