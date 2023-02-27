import pygame, math, sys
import numpy as np
import itertools
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
    def __init__(self, x_0, y_0, x_max, y_max,size, num_particles, surface):
        self.x_max = x_max
        self.y_max = y_max
        self.screen_width = x_max
        self.screen_height =y_max
        self.tile_size= size
        self.x_0 = x_0
        self.y_0 = y_0
        self.top_l = pygame.draw.line(surface, BLUE, (x_0, y_0,), (x_max, y_0))
        #self.box = pygame.draw.lines(surface, BLUE, False, ((x_0, y_0) ,(x_0, y_max),(x_max, y_0), (x_max, y_max)) , width = 2)
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

        # Add the tiles that contain more than one circle to the set of colliding tiles
        for tile_index, circle_indices in tile_circles.items():
            if len(circle_indices) > 1:
                colliding_tiles.add(tile_index)

        return colliding_tiles, tile_circles
'''
    def get_colliding_tiles(self, part_l):
        # Create a set to store tiles that contain more than one circle
        colliding_tiles = set()

        # Create a dictionary to store the circle indices for each tile
        tile_circles = {}

        # Loop through all tiles and add the circle indices to the dictionary for each tile
        for i, tile in enumerate(self.tiles):
            # Find the circles that intersect with the tile
            intersecting_circles = [j for j, part in enumerate(part_l) if part.circ.colliderect(tile)]
            if len(intersecting_circles) > 1:
                colliding_tiles.add(i)
                tile_circles[i] = intersecting_circles

        return colliding_tiles, tile_circles
'''

class Particle():
    def __init__(self, posx, posy, R,surface):
        self.pos = (posx, posy)
        self.vx = np.random.normal(0, 1)
        #self.vx = -0.2
        #self.vy = 0
        self.vy = np.random.normal(0, 1)
        self.v = (self.vy, self.vx)
        self.R = R 
        self.m = np.random.randint(1,20)
        #self.col = (np.random.randint(1,250),np.random.randint(1,250),np.random.randint(1,250))
        #print(self.vx,self.vy)
        self.col = (abs(int(self.vx*50)), abs(int(self.vy*50)), self.m)
        #self.circ = pygame.draw.circle(surface, RED, self.pos, 3)
        self.circ = pygame.draw.circle(surface, self.col, (posx,posy), R)
        #pygame.draw.rect(fake_screen, GREEN, self.shape)
    def set_vel(self, vx, vy):
        self.vx = vx
        self.vy = vy
        self.v = (self.vx, self.vy)
    def step(self ,x_0, y_0, x_max, y_max,surface):
        #rand_posx = np.random.uniform(self.circ.left-10,self.circ.left+10)
        #rand_posy = np.random.uniform(self.circ.top-10,self.circ.top+10)
        #rand_posx = np.random.choice([self.circ.left-10,self.circ.left+10, self.circ.left])
        #rand_posy = np.random.choice([self.circ.top-50,self.circ.top+10, self.circ.top])
        new_pos_x = self.pos[0] + self.vx
        new_pos_y = self.pos[1] + self.vy       
        if new_pos_x+self.R > x_max:
            self.vx *= -1
            new_pos_x = x_max - self.R
        elif new_pos_x-self.R < x_0:
            self.vx *= -1
            new_pos_x = x_0 + self.R
        if new_pos_y+self.R > y_max:
            self.vy *= -1
            new_pos_y = y_max - self.R
        elif new_pos_y-self.R < y_0:
            self.vy *= -1
            new_pos_y = y_0 + self.R

        self.pos = (new_pos_x, new_pos_y)
        
        #self.circ = self.circ.move(new_pos_x, new_pos_y)
        self.circ = pygame.draw.circle(surface, self.col, (new_pos_x, new_pos_y), self.R)
    def collides_with(self, other):
        distance = math.sqrt((self.pos[0] - other.pos[0])**2 + (self.pos[1] - other.pos[1])**2)
        return distance <= self.R + other.R
    
class Sim():
    def __init__(self,x_0, y_0, x_max, y_max,size, num_particles):
        pygame.init()
        X = 900  # screen width
        Y = 900  # screen 
        self.x_0 = x_0
        self.y_0 = y_0
        self.x_max = x_max
        self.y_max = y_max
        self.surface = pygame.display.set_mode((X, Y))
        self.surface.fill(BACKGROUND)
        self.T_l = ((x_0, y_0), (x_max, y_0))
        self.R_l = ((x_max, y_0), (x_max, y_max))
        self.L_l = ((x_0, y_0), (x_0, y_max))
        self.B_l = ((x_0, y_max), (x_max, y_max))
        
        #self.fake_screen = self.surface.copy()
        self.Radius_max = 20
        self.box = Box(0,0,900,900,20,2,self.surface)
        #self.part_l = [Particle(np.random.randint(x_0+self.Radius_max,x_max-self.Radius_max), np.random.randint(y_0+self.Radius_max,y_max-self.Radius_max), np.random.randint(10,self.Radius_max), self.surface) for i in range(num_particles)]
        self.part_l = []
        radii = []
        
        for i in range(num_particles):
            #print('nuts')
            r = np.random.randint(3, self.Radius_max)
            if any(abs(r - x) < x_max//num_particles for x in radii):
                r = np.random.randint(3, self.Radius_max)
            radii.append(r)
            
            x = np.random.randint(x_0+r+1, x_max-r-1)
            y = np.random.randint(y_0+r+1, y_max-r-1)
            self.part_l.append(Particle(x, y, r, self.surface))
            #print(x,y,r)
    def step(self):
        fake_screen = self.surface.copy()
        fake_screen.fill(BACKGROUND)
        pygame.draw.line(fake_screen, BLUE, self.T_l[0], self.T_l[1], width = 10) 
        pygame.draw.line(fake_screen, BLUE, self.R_l[0], self.R_l[1], width = 10) 
        pygame.draw.line(fake_screen, BLUE, self.L_l[0], self.L_l[1], width = 10) 
        pygame.draw.line(fake_screen, BLUE, self.B_l[0], self.B_l[1], width = 10) 
        for part in self.part_l:
            part.step(self.x_0, self.y_0, self.x_max, self.y_max,fake_screen)
        test_1, test_2 = self.box.get_colliding_tiles(self.part_l)
        colls_to_c = self.check_colls(test_2)
        #print(test_2)
        if colls_to_c != set():
            for i in colls_to_c:
                #print(self.part_l[i[0]])
                #print(self.part_l[i[1]])
                if self.part_l[i[0]].collides_with(self.part_l[i[1]]):
                    vx1f, vy1f, vx2f, vy2f, = self.collision(self.part_l[i[0]], self.part_l[i[1]])
                    self.part_l[i[0]].set_vel(vx1f, vy1f)
                    self.part_l[i[1]].set_vel(vx2f, vy2f)
                    self.part_l[i[0]].step
                    self.part_l[i[1]].step


            #print(type(colls_to_c))
        #for i in test_2.values():
        #    if self.part_l[i[0]].circ.colliderect[self.part_l[i]]
        self.surface.blit(fake_screen, (0, 0))
        
        #self.seed.step(self.surface)
        pygame.display.flip()
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
    def collision(self,part1, part2):
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
        vx1f = (v1ix * (m1 - m2) + 2 * m2 * v2ix + m1 * v1iy ** 2 - m2 * v2iy ** 2) / (m1 + m2)
        vy1f = (v1iy * (m1 - m2) + 2 * m2 * v2iy + m1 * v1ix * v1iy - m2 * v2ix * v2iy) / (m1 + m2)
        vx2f = (v2ix * (m2 - m1) + 2 * m1 * v1ix + m2 * v2iy ** 2 - m1 * v1iy ** 2) / (m1 + m2)
        vy2f = (v2iy * (m2 - m1) + 2 * m1 * v1iy + m2 * v2ix * v2iy - m1 * v1ix * v1iy) / (m1 + m2)
        #print(vx1f, vy1f, vx2f, vy2f)
         # Normalize the final velocities to ensure that the maximum velocity is 1.0
        max_vel = max(abs(vx1f), abs(vy1f), abs(vx2f), abs(vy2f))
        if max_vel > 0.0:
            vx1f /= max_vel
            vy1f /= max_vel
            vx2f /= max_vel
            vy2f /= max_vel
        
        # Calculate the final velocities' magnitudes and directions
        #v1f = math.sqrt(max(vx1f ** 2 + vy1f ** 2, 1e-10))
        #v2f = math.sqrt(max(vx2f ** 2 + vy2f ** 2, 1e-10))
        #theta1f = math.atan2(vy1f, vx1f)
        #theta2f = math.atan2(vy2f, vx2f)
     
        new_pos1x = part1.pos[0] + vx1f
        new_pos1y = part1.pos[1] + vy1f
        new_pos2x = part2.pos[0] + vx2f
        new_pos2y = part2.pos[1] + vy2f
        # Check if the particles are still overlapping and move them apart if necessary
        dist = math.sqrt((new_pos1x - new_pos2x) ** 2 + (new_pos1y - new_pos2y) ** 2)
        overlap = (part1.R + part2.R) - dist
        pos1_list = list(part1.pos)
        pos2_list = list(part2.pos)
        
        if overlap > 0:
            nudge = overlap*1.2
            
            # Move the particles apart in proportion to their masses
            total_mass = m1 + m2
            new_pos1x += nudge * (m2 / total_mass) * (new_pos1x - new_pos2x) / dist
            new_pos1y += nudge * (m2 / total_mass) * (new_pos1y - new_pos2y) / dist
            new_pos2x -= nudge * (m1 / total_mass) * (new_pos1x - new_pos2x) / dist
            new_pos2y -= nudge * (m1 / total_mass) * (new_pos1y - new_pos2y) / dist
            part1.pos = (new_pos1x, new_pos1y)
            part2.pos = (new_pos2x, new_pos2y)
            part1.circ = pygame.draw.circle(self.surface, part1.col, (new_pos1x, new_pos1y),part1.R)
            part2.circ = pygame.draw.circle(self.surface, part2.col, (new_pos2x, new_pos2y), part2.R)
            #dist = math.sqrt((new_pos1x - new_pos2x) ** 2 + (new_pos1y - new_pos2y) ** 2)
            #overlap = (part1.R + part2.R) - dist
            #if overlap > 0:
            # Reflect velocities and add a small random nudge to escape overalps
            #vx1f += np.random.randint(1,2)/100
            #vy1f += np.random.randint(1,2)/100
            #vx2f += np.random.randint(1,2)/100
            #vy2f += np.random.randint(1,2)/100
            #vx1f *= -1
            #vy1f *= -1
            #vx2f *= -1
            #vy2f *= -1
            #vx1f += np.random.randint(20,30)/100
            ##vy1f += np.random.randint(20,30)/100
            #vx2f += np.random.randint(20,30)/100
            #vy2f += np.random.randint(20,30)/100
            #vx1f += np.random.normal(0,1)
            #vy1f += np.random.normal(0,1)
            #vx2f += np.random.normal(0,1)
            #vy2f += np.random.normal(0,1)
            

        return vx1f, vy1f, vx2f, vy2f
    

simulation = Sim(0,0,900,900,1,500)
clock =pygame.time.Clock()

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
    pygame.display.flip()
    simulation.step()
    clock.tick(120)
