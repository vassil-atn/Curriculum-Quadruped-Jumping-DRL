# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import numpy as np
from numpy.random import choice
from scipy import interpolate

from isaacgym import terrain_utils
from isaacgym.terrain_utils import *
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg

class Terrain:
    def __init__(self, cfg: LeggedRobotCfg.terrain, num_robots) -> None:

        self.cfg = cfg
        self.num_robots = num_robots
        self.type = cfg.mesh_type
        if self.type in ["none", 'plane']:
            return
        self.env_length = cfg.terrain_length
        self.env_width = cfg.terrain_width
        self.proportions = [np.sum(cfg.terrain_proportions[:i+1]) for i in range(len(cfg.terrain_proportions))]

        self.cfg.num_sub_terrains = cfg.num_rows * cfg.num_cols
        self.env_origins = np.zeros((cfg.num_rows, cfg.num_cols, 3))
        self.env_properties = np.zeros((cfg.num_rows, cfg.num_cols, 3))

        self.width_per_env_pixels = int(self.env_width / cfg.horizontal_scale)
        self.length_per_env_pixels = int(self.env_length / cfg.horizontal_scale)

        self.border = int(cfg.border_size/self.cfg.horizontal_scale)
        self.tot_cols = int(cfg.num_cols * self.width_per_env_pixels) + 2 * self.border
        self.tot_rows = int(cfg.num_rows * self.length_per_env_pixels) + 2 * self.border

        self.height_field_raw = np.zeros((self.tot_rows , self.tot_cols), dtype=np.int16)
        if cfg.selected and cfg.curriculum:
            self.selected_terrain_curriculum()
        elif cfg.curriculum and not cfg.selected:
            self.curiculum()
        elif cfg.selected:
            self.selected_terrain()
        else:    
            self.flat_terrain()
            # self.randomized_terrain()   
        
        self.heightsamples = self.height_field_raw
        if self.type=="trimesh":
            self.vertices, self.triangles = terrain_utils.convert_heightfield_to_trimesh(   self.height_field_raw,
                                                                                            self.cfg.horizontal_scale,
                                                                                            self.cfg.vertical_scale,
                                                                                            self.cfg.slope_treshold)
    
    def flat_terrain(self):
        for k in range(self.cfg.num_sub_terrains):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))

            terrain = terrain_utils.SubTerrain("terrain",
                              width=self.width_per_env_pixels,
                              length=self.width_per_env_pixels,
                              vertical_scale=self.cfg.vertical_scale,
                              horizontal_scale=self.cfg.horizontal_scale)
            if self.cfg.make_terrain_uneven:
                terrain.height_field_raw = terrain_utils.random_uniform_terrain(terrain, min_height=-0.03, max_height=0.03, step=0.01, downsampled_scale=0.2).height_field_raw

            self.add_terrain_to_map(terrain, i, j)

    def randomized_terrain(self):
        for k in range(self.cfg.num_sub_terrains):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))

            choice = np.random.uniform(0, 1)
            difficulty = np.random.choice([0.5, 0.75, 0.9])
            terrain = self.make_terrain(choice, difficulty)
            self.add_terrain_to_map(terrain, i, j)
    


    def selected_terrain_curriculum(self):
        terrain_type = self.cfg.terrain_kwargs.pop('type')
        sloped_terrain_number = self.cfg.sloped_terrain_number
        if self.cfg.make_terrain_uneven and 'make_uneven' in self.cfg.terrain_kwargs:
            self.cfg.terrain_kwargs['make_uneven'] = True

        kwargs = self.cfg.terrain_kwargs
        # The first x terrains should be just flat ground.
        terrain_difficulty_height_zero = np.zeros(self.cfg.num_zero_height_terrains)
        

        terrain_difficulty_height_rest = np.linspace(self.cfg.terrain_difficulty_height_range[0], self.cfg.terrain_difficulty_height_range[1], self.cfg.num_rows - self.cfg.num_zero_height_terrains)
        
        terrain_difficulty_height = np.hstack((terrain_difficulty_height_zero,terrain_difficulty_height_rest))

        terrain_difficulty_width = np.linspace(self.cfg.terrain_difficulty_width_range[0], self.cfg.terrain_difficulty_width_range[1], self.cfg.num_cols)
        
        terrain_difficulty_width[(terrain_difficulty_width>0.2) * (terrain_difficulty_width<0.6)] = 0.6
        
        terrain_lengths = np.random.uniform(0.8, 1.6, (self.cfg.num_cols,self.cfg.num_rows))

        slopes = np.linspace(self.cfg.slope_range[0], self.cfg.slope_range[1], self.cfg.num_rows- self.cfg.num_zero_height_terrains)
        terrain_difficulty_slopes = np.hstack((terrain_difficulty_height_zero,slopes))
        
        for j in range(self.cfg.num_cols - sloped_terrain_number):
            kwargs["box_width"] = terrain_difficulty_width[j]
            
            for i in range(self.cfg.num_rows):
                kwargs["box_height"] = terrain_difficulty_height[i]


                terrain = terrain_utils.SubTerrain("terrain",
                            width=self.width_per_env_pixels,
                            length=self.width_per_env_pixels,
                            vertical_scale=self.cfg.vertical_scale,
                            horizontal_scale=self.cfg.horizontal_scale)
                
                kwargs["box_length"] = np.clip(terrain_lengths[j,i], a_min= 0.0, a_max = terrain.length * terrain.horizontal_scale)

                eval(terrain_type)(terrain, **self.cfg.terrain_kwargs)

                self.add_terrain_to_map(terrain, i, j)
                self.env_properties[i,j,0] = kwargs["box_width"]
                self.env_properties[i,j,1] = kwargs["box_length"]
                self.env_properties[i,j,2] = kwargs["box_height"]   
        
        # Sloped terrain

        for j in range(self.cfg.num_cols-sloped_terrain_number, self.cfg.num_cols):
            for i in range(self.cfg.num_rows):

                box_length = np.clip(terrain_lengths[j,i], a_min= 0.0, a_max = terrain.length * terrain.horizontal_scale)

                terrain = terrain_utils.SubTerrain("terrain",
                width=self.width_per_env_pixels,
                length=self.width_per_env_pixels,
                vertical_scale=self.cfg.vertical_scale,
                horizontal_scale=self.cfg.horizontal_scale)
                

                platform_size = 1.
                slope = terrain_difficulty_slopes[i]
                height = slope*platform_size

                if slope == 0:
                    continue # check is this messes up the env props

                partial_sloped_terrain(terrain,slope=slope,width=platform_size, length = box_length)
                self.add_terrain_to_map(terrain, i, j)
                self.env_properties[i,j,0] = platform_size # CHECK THESE
                self.env_properties[i,j,1] = box_length # CHECK THESE
                self.env_properties[i,j,2] = height


        # For the flat terrains set the env properties to 0 (no obstacles)
        

        self.env_properties[terrain_difficulty_height==0, :, :] = 0

    def curiculum(self):
        for j in range(self.cfg.num_cols):
            for i in range(self.cfg.num_rows):
                difficulty = i / self.cfg.num_rows
                choice = j / self.cfg.num_cols + 0.001

                terrain = self.make_terrain(choice, difficulty)
                self.add_terrain_to_map(terrain, i, j)

    def selected_terrain(self):
        terrain_type = self.cfg.terrain_kwargs.pop('type')
        for k in range(self.cfg.num_sub_terrains):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))

            terrain = terrain_utils.SubTerrain("terrain",
                              width=self.width_per_env_pixels,
                              length=self.width_per_env_pixels,
                              vertical_scale=self.cfg.vertical_scale,
                              horizontal_scale=self.cfg.horizontal_scale)

            eval(terrain_type)(terrain, **self.cfg.terrain_kwargs)
            self.add_terrain_to_map(terrain, i, j)
    
    def make_terrain(self, choice, difficulty):
        terrain = terrain_utils.SubTerrain(   "terrain",
                                width=self.width_per_env_pixels,
                                length=self.width_per_env_pixels,
                                vertical_scale=self.cfg.vertical_scale,
                                horizontal_scale=self.cfg.horizontal_scale)
        slope = difficulty * 0.4
        step_height = 0.05 + 0.18 * difficulty
        discrete_obstacles_height = 0.05 + difficulty * 0.2
        stepping_stones_size = 1.5 * (1.05 - difficulty)
        stone_distance = 0.05 if difficulty==0 else 0.1
        gap_size = 1. * difficulty
        pit_depth = 1. * difficulty
        if choice < self.proportions[0]:
            if choice < self.proportions[0]/ 2:
                slope *= -1
            terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)
        elif choice < self.proportions[1]:
            terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)
            terrain_utils.random_uniform_terrain(terrain, min_height=-0.05, max_height=0.05, step=0.005, downsampled_scale=0.2)
        elif choice < self.proportions[3]:
            if choice<self.proportions[2]:
                step_height *= -1
            terrain_utils.pyramid_stairs_terrain(terrain, step_width=0.31, step_height=step_height, platform_size=3.)
        elif choice < self.proportions[4]:
            num_rectangles = 20
            rectangle_min_size = 1.
            rectangle_max_size = 2.
            terrain_utils.discrete_obstacles_terrain(terrain, discrete_obstacles_height, rectangle_min_size, rectangle_max_size, num_rectangles, platform_size=3.)
        elif choice < self.proportions[5]:
            terrain_utils.stepping_stones_terrain(terrain, stone_size=stepping_stones_size, stone_distance=stone_distance, max_height=0., platform_size=4.)
        elif choice < self.proportions[6]:
            gap_terrain(terrain, gap_size=gap_size, platform_size=3.)
        else:
            pit_terrain(terrain, depth=pit_depth, platform_size=4.)
        
        return terrain

    def add_terrain_to_map(self, terrain, row, col):
        i = row
        j = col
        # map coordinate system
        start_x = self.border + i * self.length_per_env_pixels
        end_x = self.border + (i + 1) * self.length_per_env_pixels
        start_y = self.border + j * self.width_per_env_pixels
        end_y = self.border + (j + 1) * self.width_per_env_pixels
        self.height_field_raw[start_x: end_x, start_y:end_y] = terrain.height_field_raw

        env_origin_x = (i + 0.5) * self.env_length
        env_origin_y = (j + 0.5) * self.env_width
        x1 = int((self.env_length/2. - 1) / terrain.horizontal_scale)
        x2 = int((self.env_length/2. + 1) / terrain.horizontal_scale)
        y1 = int((self.env_width/2. - 1) / terrain.horizontal_scale)
        y2 = int((self.env_width/2. + 1) / terrain.horizontal_scale)
        env_origin_z = np.max(terrain.height_field_raw[x1:x2, y1:y2])*terrain.vertical_scale
        self.env_origins[i, j] = [env_origin_x, env_origin_y, env_origin_z]


def gap_terrain(terrain, gap_size, platform_size=1.):
    gap_size = int(gap_size / terrain.horizontal_scale)
    platform_size = int(platform_size / terrain.horizontal_scale)

    center_x = terrain.length // 2
    center_y = terrain.width // 2
    x1 = (terrain.length - platform_size) // 2
    x2 = x1 + gap_size
    y1 = (terrain.width - platform_size) // 2
    y2 = y1 + gap_size
   
    terrain.height_field_raw[center_x-x2 : center_x + x2, center_y-y2 : center_y + y2] = -1000
    terrain.height_field_raw[center_x-x1 : center_x + x1, center_y-y1 : center_y + y1] = 0

def pit_terrain(terrain, depth, platform_size=1.):
    depth = int(depth / terrain.vertical_scale)
    platform_size = int(platform_size / terrain.horizontal_scale / 2)
    x1 = terrain.length // 2 - platform_size
    x2 = terrain.length // 2 + platform_size
    y1 = terrain.width // 2 - platform_size
    y2 = terrain.width // 2 + platform_size
    terrain.height_field_raw[x1:x2, y1:y2] = -depth




def box_terrain(terrain, box_width, box_length, box_height, make_uneven=False):
    """
    Generate a box

    Parameters:
        terrain (terrain): the terrain
        step_width (float):  the width of the step [meters]
        step_height (float):  the height of the step [meters]
    Returns:
        terrain (SubTerrain): update terrain
    """
    # switch parameters to discrete units
    box_width = int(round(box_width / terrain.horizontal_scale))
    box_height = int(round(box_height / terrain.vertical_scale))
    box_length = int(round(box_length / terrain.horizontal_scale))

    num_steps = terrain.width // box_width
    height = box_height
    i = int(num_steps/2)
    
    center_x = terrain.width // 2
    center_y = terrain.length // 2

    x1 = 0
    x2 = x1 + int(box_width/2)
    y1 = 0
    y2 = y1 + int(box_length/2)


    # terrain.height_field_raw[i * box_width: (i + 1) * box_width, :] += height
    terrain.height_field_raw[center_x-x2 : center_x + x2, center_y-y2 : center_y + y2] += height

    if make_uneven:
        terrain.height_field_raw += terrain_utils.random_uniform_terrain(terrain, min_height=-0.02, max_height=0.02, step=0.01).height_field_raw
        
    return terrain


def partial_sloped_terrain(terrain, slope=1,width=None,length=None):
    """
    Generate a sloped terrain

    Parameters:
        terrain (SubTerrain): the terrain
        slope (int): positive or negative slope
    Returns:
        terrain (SubTerrain): update terrain
    """

    center_x = terrain.width // 2
    center_y = terrain.length // 2

    box_half_width = int(round((width / terrain.horizontal_scale) / 2))
    box_half_length = int(round((length / terrain.horizontal_scale) / 2))

    x = np.arange(0, 2*box_half_width)
    y = np.arange(0, 2*box_half_length)

    xx, yy = np.meshgrid(x, y, sparse=True)
    xx = xx.reshape(len(x), 1)

    max_height = int(slope * (terrain.horizontal_scale / terrain.vertical_scale) * terrain.width)
    terrain.height_field_raw[center_x - box_half_width : center_x + box_half_width, 
                             center_y - box_half_length : center_y + box_half_length] += (max_height * xx / len(x)).astype(terrain.height_field_raw.dtype)
    
    obj_width = int(width/terrain.horizontal_scale)

    terrain.height_field_raw[np.arange(obj_width,len(x)), :] = 0

    return terrain