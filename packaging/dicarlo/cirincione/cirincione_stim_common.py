
import numpy as np
import imageio
import os
import pandas as pd
from glob import glob
import scipy.ndimage as nd
import matplotlib.pyplot as plt
from brainio.stimuli import StimulusSet
from tqdm import tqdm

xyY_COLOR_DICT = {'red': np.array([.6, .35, 14]),
                  'brown': np.array([.6, .35, 2.7]),
                  'green': np.array([.31, .58, 37]),
                  'olive': np.array([.31, .58, 6.7]),
                  'blue': np.array([.16, .08, 6.8]),
                  'azure': np.array([.16, .08, 1.8]),
                  'yellow': np.array([.41, .50, 37]),
                  'beige': np.array([.46, .45, 6.5]),
                  'violet': np.array([.3, .15, 20]),
                  'purple': np.array([.3, .15, 3.4]),
                  'aqua': np.array([.23, .31, 38]),
                  'cyan': np.array([.23, .31, 7.3]),
                  'white': np.array([.3, .32, 38]),
                  'gray': np.array([.3, .32, 8.8]),
                  'black': np.array([.3, .32, 1.2]),
                  'light_gray': np.array([.3, .32, 20])}

class BO_Stimulus:
    def __init__(self, save_dir, stim_size = 672, visual_degrees = 12,
                                 divisions = 12, posy = 0.5, posx = 0.5, sqr_deg=4):
        self.save_dir = save_dir
        self.stim_size = stim_size
        self.visual_degrees = visual_degrees
        self.divisions = divisions
        self.posx = posx
        self.posy = posy
        self.sqr_deg = sqr_deg
        self.xyY_color_dict = xyY_COLOR_DICT
        self.colors = self.xyY_to_RGB()
        self.ground = self.colors['light_gray']

        BO_standard_test_stim_data = pd.DataFrame(
            columns=['image_id', 'degrees', 'posy', 'posx', 'color', 'orientation', 'polarity', 'side'])
        DIR = self.save_dir
        color_idx = 0
        print('Generating Stimulus...')
        for color_name in tqdm(self.colors.keys()[0:14]):

            for polarity in range(2):
                if polarity == 0:
                    ground = self.ground
                    figure = self.colors[color_name]
                else:
                    ground = self.colors[color_name]
                    figure = self.ground

                for side in range(2):
                    if side == 0:
                        xshift = 0.5
                    else:
                        xshift = -0.5
                    BO_standard_test_stim_oris = self.generate_bar_stim(length=self.sqr_deg / self.visual_degrees,
                                                                   width=self.sqr_deg / self.visual_degrees,
                                                                   stim_size=self.stim_size,
                                                                   divisions=self.divisions, figure_color=figure,
                                                                   ground_color=ground, xshift=xshift,
                                                                   posx=self.posx / self.visual_degrees,
                                                                   posy=self.posy / self.visual_degrees)
                    division_idx = 0
                    for d in range(divisions):
                        ID = str(color_idx).zfill(2) + str(polarity).zfill(2) + str(side).zfill(2) + str(
                            division_idx).zfill(2)
                        BO_standard_test_stim_img = BO_standard_test_stim_oris[d].astype(np.uint8)

                        BO_standard_test_stim_data = BO_standard_test_stim_data.append(
                            {'image_id': ID, 'degrees': self.visual_degrees, 'posy': self.posy, 'posx': self.posx, 'color': color_name,
                             'orientation': 180 / 12 * d, 'polarity': polarity, 'side': side, },
                            ignore_index=True)

                        file_name = 'BO_standard_test_stim_' + str(ID) + '.png'
                        imageio.imwrite(DIR + file_name, BO_standard_test_stim_img)

                        division_idx += 1
            color_idx += 1
        BO_standard_test_stim_data.to_csv(DIR + 'stimulus_set', index=False)
        self.stim_data = BO_standard_test_stim_data

    def generate_bar_stim(self, length, width, stim_size, divisions, figure_color, ground_color, xshift, yshift, posx, posy):
        angles = np.linspace(90, -90, divisions, endpoint=False)

        xshift = int((stim_size * xshift) // 2)
        yshift = int((stim_size * yshift) // 2)
        radius_L = int((stim_size * length) // 2)
        radius_W = int((stim_size * width) // 2)

        square_stim = np.zeros((divisions, stim_size, stim_size, 3))
        surround_stim = np.zeros((divisions, stim_size * 2, stim_size * 2, 3))
        surround_stim[:, :, :] = ground_color

        origin = stim_size // 2
        x_lim = np.linspace(2 * origin - radius_W, 2 * origin + radius_W, radius_W * 2, endpoint=False) + xshift
        y_lim = np.linspace(2 * origin - radius_L, 2 * origin + radius_L, radius_L * 2, endpoint=False) + yshift

        idx = 0
        for ang in (angles):
            for x in x_lim:
                for y in y_lim:
                    col = figure_color
                    try:
                        surround_stim[idx, int(np.floor(y)), int(np.floor(x))] = col
                    except:
                        pass
            surround_rotated = nd.rotate(surround_stim[idx], ang, reshape=False, prefilter=False)
            square_stim[idx] = surround_rotated[int((stim_size - 0.5 * stim_size) - posy * stim_size):int(
                (stim_size + 0.5 * stim_size) - posy * stim_size),
                               int((stim_size - 0.5 * stim_size) - posx * stim_size):int(
                                   (stim_size + 0.5 * stim_size) - posx * stim_size)]
            idx += 1
        return square_stim.astype(int)


    def xyY_to_RGB(self):
        #Credit to https://www.easyrgb.com/en/math.php
        xyY_color_dict = self.xyY_color_dict
        rgb_color_dict = pd.DataFrame(columns=[xyY_color_dict.keys()])  # posx and posy offset
        for keys in xyY_color_dict.keys():
            Y = xyY_color_dict[keys][2]
            X = xyY_color_dict[keys][0] * (Y / xyY_color_dict[keys][1])
            Z = (1 - xyY_color_dict[keys][0] - xyY_color_dict[keys][1]) * (Y / xyY_color_dict[keys][1])

            var_X = X / 100
            var_Y = Y / 100
            var_Z = Z / 100

            var_R = var_X * 3.2406 + var_Y * -1.5372 + var_Z * -0.4986
            var_G = var_X * -0.9689 + var_Y * 1.8758 + var_Z * 0.0415
            var_B = var_X * 0.0557 + var_Y * -0.2040 + var_Z * 1.0570

            if (var_R > 0.0031308):
                var_R = 1.055 * ( var_R ** ( 1 / 2.4 ) ) - 0.055
            else:
                var_R = 12.92 * var_R
            if (var_G > 0.0031308):
                var_G = 1.055 * ( var_G ** ( 1 / 2.4 ) ) - 0.055
            else:
                var_G = 12.92 * var_G
            if (var_B > 0.0031308):
                var_B = 1.055 * ( var_B ** ( 1 / 2.4 ) ) - 0.055
            else:
                var_B = 12.92 * var_B

            sR = var_R * 255
            sG = var_G * 255
            sB = var_B * 255

            rgb_color_dict[keys] = np.array([sR, sG, sB])
        return rgb_color_dict

def load_stim_info(stim_name, data_dir):
    stim = pd.read_csv(os.path.join(data_dir, 'stimulus_set'), dtype={'image_id': str})
    image_paths = dict((key, value) for (key, value) in zip(stim['image_id'].values,
                                                            [os.path.join(data_dir, image_name) for image_name
                                                             in stim['image_file_name'].values]))
    stim_set = StimulusSet(stim[stim.columns[:-1]])
    stim_set.image_paths = image_paths
    stim_set.identifier = stim_name

    return stim_set

def gen_BO_stim(BO_params, save_dir):
    if not (os.path.isdir(save_dir)):
        os.mkdir(save_dir)
   # BO_PARAMS = np.array([DEGREES, SIZE_PX, ORIENTATION_DIV,
    # POS_X, POS_Y, SQUARE_SIZE])

    BO_stim = BO_Stimulus(save_dir=save_dir,
                          visual_degrees=BO_params[0],
                          stim_size=BO_params[1],
                          divisions=BO_params[2],
                          posx=BO_params[3],
                          posy=BO_params[4],
                          sqr_deg=BO_params[5])

    return