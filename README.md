# Mutliparticle_Model
Simple elastic Collision Model working on pygame that can handle several hundred particles at a time. Uses Uniform grid partitioning to find collisions between particles as an optimization. To run the code, simply clone and travel to this repository and run the command:
"python3 Elastic_collision_sim.py" 
To edit the parameters of the model, edit line 255, giving the Sim object as follows "Sim(screen_min_x, screen_min_y, screen_max_x,screen_max_y, tile_size, Num_particles)"
Tile size is the size of each grid used for partitioning, the larger the tile size, the smaller the total grid.