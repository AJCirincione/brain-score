# Packages. I'll cover what they do when we run across them.
import numpy as np
import imageio
import os
import pandas as pd
from glob import glob
import scipy.ndimage as nd
import matplotlib.pyplot as plt
from brainio.stimuli import StimulusSet
from tqdm import tqdm

# xyY Color Dictionary (x and y are chromaticity coordinates, Y is luminance)
# All these values are taken directly from the zhou et al. (2000) border ownership paper
# The final color, light_gray, is used as ground color for all experiments
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

# This is setting up a dictionary with all the same colors as keys but it will eventually
# hold RGB values corresponding to the xyY dictionary above after we convert.
blank = {'red': None,
                  'brown': None,
                  'green': None,
                  'olive': None,
                  'blue': None,
                  'azure': None,
                  'yellow': None,
                  'beige': None,
                  'violet': None,
                  'purple': None,
                  'aqua': None,
                  'cyan': None,
                  'white': None,
                  'gray': None,
                  'black': None,
                  'light_gray': None}

# This first class creates the optimization set - ie, bars at different lengths, widths,
# colors, and orientations. Neither this nor the following class actually have to be classes,
# everything is done in the initialize function, so as of now having it be a class is pointless - I just wanted to
# remain consistent with previous formats for stim_common on here.
class BO_Optim:
    def __init__(self, save_dir, stim_size = 672, visual_degrees = 12,
                                 divisions = 12, posy = 0.5, posx = 0.5):

        # The directory in which the stimulus is saved
        self.save_dir = save_dir

        # The total size of each image, corresponding to its visual degrees (12 degs = 672 px)
        self.stim_size = int(stim_size)

        # The visual degrees each image is supposed to take up
        self.visual_degrees = visual_degrees

        # How many steps are there in different orientations from 0 to 180? If divisions is 12,
        # each step is 15 degs as 180/12 = 15.
        self.divisions = int(divisions)

        # Slight 0.5 deg offset to study neurons that are not directly in the center of the visual field.
        self.posx = posx
        self.posy = posy

        # uses xyY_to_RGB function to convert original xyY dictionary to RGB
        self.colors = xyY_to_RGB(xyY_COLOR_DICT, blank)

        # The ground color of all experiments
        self.ground = self.colors['light_gray']

        # Varying width and length of stimulus
        width = [0.1, 0.2]
        length = [0.75, 1.5]

        # This data frame contains all information about the stimulus, on top of its properties it also includes
        # the ID, the file name and the file path.
        # This will be converted to csv at the end.
        BO_optim_stim_data = pd.DataFrame(
            columns=['stimulus_id', 'degrees', 'position_y', 'position_x', 'color', 'orientation', 'width',
                     'length', 'image_file_name', 'image_current_local_file_path'])

        # Variable that will be used to encode color in the image_id (00 is red, 01 is brown, etc). It will count up
        # as the for loop loops through the colors. Similar will be done for other properties.
        color_idx = 0


        print('Constructing...')
        # Note we loop through 15 colors ([0:14]) and not the last. The last color is always the ground color, and
        # in this set is never assigned to the figure.
        for color_name in tqdm(list(self.colors.keys())[0:15]):

            # Used to encode width as either 00 (0.1) or 01 (0.2)
            width_idx = 0


            for W in width:

                # Used to encode length as either 00 (0.75) or 01 (1.5)
                length_idx = 0

                for L in length:

                    # Calls upon generate_bar_stim function. This function returns bars of desired length, width, etc at
                    # all orientations. If you specified 12 divisions, the functions would return bars at 0 degs,
                    # 15 degs, 30 degs, ..., 165 degs.
                    BO_optim_stim_oris = generate_bar_stim(length=L, width=W,
                                                           stim_size=stim_size,
                                                           divisions=self.divisions, figure_color=self.colors[color_name],
                                                           ground_color=self.ground,
                                                           posx=posx,
                                                           posy=posy)

                    # Used to encode orientation division. 0 degs is 00 and following orientations are 01, 02, etc...
                    division_idx = 0

                    for d in range(self.divisions):

                        # Save each property in two digit spaces. First two digits is color, then width, then length,
                        # and finally orientation division. For example, 00000000 would be a red stimulus with
                        # 0.1 width and 0.75 length at 0 degs. 00000001 would be the same, but at 15 degs.
                        # 01000001 would be the same as 00000001, but brown, etc...
                        ID = str(color_idx).zfill(2) + str(width_idx).zfill(2) + str(length_idx).zfill(2) + str(
                            division_idx).zfill(2)

                        # Convert stimulus to uint8 format (saves on memory?)
                        BO_optim_stim_img = BO_optim_stim_oris[d].astype(np.uint8)

                        # File name, which is BO_optim_<ID>.png
                        file_name = 'BO_optim_' + str(ID) + '.png'

                        # Add all properties of the saved stimulus as a new row in the data frame initialized earlier.
                        BO_optim_stim_data = BO_optim_stim_data.append(
                            {'stimulus_id': ID, 'degrees': visual_degrees, 'position_y': posy, 'position_x': posx, 'color': color_name,
                             'orientation': 180 / self.divisions * d, 'width': W, 'length': L, 'image_file_name': file_name, 'image_current_local_file_path': self.save_dir},
                            ignore_index=True)

                        # Save image to save_dir (save directory)
                        imageio.imwrite(os.path.join(self.save_dir,file_name), BO_optim_stim_img)

                        # Move to next division id
                        division_idx += 1

                    # Move to next length id, etc...
                    length_idx += 1
                width_idx += 1
            color_idx += 1

        # Save dataframe as csv file in same file path as images.
        BO_optim_stim_data.to_csv(os.path.join(self.save_dir,'stimulus_set'), index=False)
        self.stim_data = BO_optim_stim_data

# Class that creates stimulus for the standard test. Very similar to a lot of what BO_optim does, so I will just
# comment when something new comes up.
class BO_Stimulus:
    def __init__(self, save_dir, stim_size = 672, visual_degrees = 12,
                                 divisions = 12, posy = 0.5, posx = 0.5, sqr_deg=4):
        self.save_dir = save_dir
        self.stim_size = int(stim_size)
        self.blank = blank
        self.visual_degrees = visual_degrees
        self.divisions = int(divisions)
        self.posx = posx
        self.posy = posy

        # Degrees of the square in the standard test. This should always be 4 degrees.
        self.sqr_deg = int(sqr_deg)

        self.xyY_color_dict = xyY_COLOR_DICT
        self.colors = xyY_to_RGB(xyY_COLOR_DICT, blank)
        self.ground = self.colors['light_gray']

        # Friday - have responses to 4 stimuli A,B,C,D

        # Very similar to dataframe in previous class, but now we keep track of polarity (switch ground and figure colors)
        # and side (flip square over inner side) instead of length and width (consistent in this stimulus)
        BO_standard_test_stim_data = pd.DataFrame(
            columns=['stimulus_id', 'degrees', 'position_y', 'position_x', 'color', 'orientation', 'polarity', 'side',
                     'image_file_name', 'image_current_local_file_path'])

        color_idx = 0
        print('Generating Stimulus...')
        for color_name in tqdm(list(self.colors.keys())[0:15]):

            for polarity in range(2):

                # If polarity is 0 then figure color and ground color remain the same. If polarity is 1, the colors
                # are then swapped
                if polarity == 0:
                    ground = self.ground
                    figure = self.colors[color_name]
                else:
                    ground = self.colors[color_name]
                    figure = self.ground

                for side in range(2):

                    # If side is 0 the square is shifted to the 12 o'clock position and rotated CL to the 6 o'clock
                    # position. If side is 1, the square flips to the opposite side of the 'clock'
                    if side == 0:
                        xshift = 2
                    else:
                        xshift = -2

                    # Information on xshift and yshift seen in generate_bar_stim function
                    BO_standard_test_stim_oris = generate_bar_stim(length=self.sqr_deg,
                                                                   width=self.sqr_deg,
                                                                   stim_size=self.stim_size,
                                                                   divisions=self.divisions, figure_color=figure,
                                                                   ground_color=ground, xshift=xshift,
                                                                   yshift=0,
                                                                   posx=self.posx,
                                                                   posy=self.posy,
                                                                   visual_degrees=self.visual_degrees)

                    division_idx = 0
                    for d in range(self.divisions):

                        ID = str(color_idx).zfill(2) + str(polarity).zfill(2) + str(side).zfill(2) + str(
                            division_idx).zfill(2)

                        BO_standard_test_stim_img = BO_standard_test_stim_oris[d].astype(np.uint8)

                        file_name = 'BO_stim_' + str(ID) + '.png'

                        BO_standard_test_stim_data = BO_standard_test_stim_data.append(
                            {'stimulus_id': ID, 'degrees': self.visual_degrees, 'position_y': self.posy, 'position_x': self.posx, 'color': color_name,
                             'orientation': 180 / divisions * d, 'polarity': polarity, 'side': side, 'image_file_name': file_name, 'image_current_local_file_path': self.save_dir},
                            ignore_index=True)

                        imageio.imwrite(os.path.join(self.save_dir,file_name), BO_standard_test_stim_img)

                        division_idx += 1
            color_idx += 1

        BO_standard_test_stim_data.to_csv(os.path.join(self.save_dir,'stimulus_set'), index=False)
        self.stim_data = BO_standard_test_stim_data

#This function converts from xyY to XYZ format, then XYZ to RGB format.
def xyY_to_RGB(xyY_color_dict, blank):

        #Credit to https://www.easyrgb.com/en/math.php for the functions and code.
        xyY_color_dict = xyY_color_dict
        rgb_color_dict = blank
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

# Function used to generate all stimulus
def generate_bar_stim(length, width, stim_size, divisions, figure_color, ground_color, xshift=0, yshift=0, posx=0.1,
                      posy=0.1, visual_degrees=12):

        # Amount of divisions of orientation (as before)
        divisions = int(divisions)

        # Size in px of entire stimulus (also as before)
        stim_size = int(stim_size)

        # Linspace of all orientations cut into desired divisions. We go from 90 to -90 instead of 0 to 180 as
        # the 12 o'clock position (90 degs) is considered 0 degrees experimentally.
        angles = np.linspace(90, -90, divisions, endpoint=False)

        # How much is the stimulus shifted on the x axis? The units are originally passed in degrees, then are converted
        # to (shifted degrees)/(total visual degrees) to get the proportion of the image the shift takes up,
        # and are then multiplied by the stim_size to get this value in pixels. Similar idea for posx and posy / length
        # and width
        xshift = int((stim_size * (xshift/visual_degrees)))
        yshift = int((stim_size * (yshift/visual_degrees)))
        posx = posx/visual_degrees * stim_size
        posy = posy/visual_degrees * stim_size
        length = length/visual_degrees * stim_size
        width = width/visual_degrees * stim_size

        # 'Radius' of the bar concerning length and width (really just half length and half width in px)
        radius_L = int(length // 2)
        radius_W = int(width // 2)

        # Sets up array for each stimulus at different orientations
        square_stim = np.zeros((divisions, stim_size, stim_size, 3))

        # Used while rotating stimulus. This is a larger array which is cropped after the fact. The reasoning for this
        # is if the bar in square_stim is cut off at all before it is rotated, only the cut off portion will rotate while
        # any stimulus outside the cut off area is disregarded. This is a workaround.
        surround_stim = np.zeros((divisions, stim_size * 2, stim_size * 2, 3))

        #Set everything as ground color initially, figure color assigned in a bit
        surround_stim[:, :, :] = ground_color

        #Sets origin as well as x_lim and y_lim - the space the figure / bar stimulus takes up.
        origin = stim_size // 2
        x_lim = np.linspace(2 * origin - radius_W, 2 * origin + radius_W, radius_W * 2, endpoint=False) + xshift
        y_lim = np.linspace(2 * origin - radius_L, 2 * origin + radius_L, radius_L * 2, endpoint=False) + yshift

        idx = 0
        # Loop through all orientations
        for ang in (angles):

            # Contained within x_lim
            for x in x_lim:

                # Contained with y_lim
                for y in y_lim:

                    # We use 'try' here is there is a chance the area that we are trying to color in for the figure goes
                    # outside array dimensions. If it does, nothing happens as opposed to an error being thrown
                    try:
                        surround_stim[idx, int(np.floor(y)), int(np.floor(x))] = figure_color
                    except:
                        pass

            # Rotate stimulus after figure has been filled with color
            surround_rotated = nd.rotate(surround_stim[idx], ang, reshape=False, prefilter=False)

            # Crop the 'surround' to get the desired size for the original stimulus. Shift by posx and posy.
            square_stim[idx] = surround_rotated[int((stim_size - 0.5 * stim_size) - posy):int(
                (stim_size + 0.5 * stim_size) - posy),
                               int((stim_size - 0.5 * stim_size) - posx):int(
                                   (stim_size + 0.5 * stim_size) - posx)]
            idx += 1

        return square_stim.astype(int)



def load_stim_info(stim_name, data_dir):
    # Loads csv
    stim = pd.read_csv(os.path.join(data_dir, 'stimulus_set'), dtype={'stimulus_id': str})

    # Gives dictionary of image_id's and respective file paths
    image_paths = dict((key, value) for (key, value) in zip(stim['stimulus_id'].values,
                                                            [os.path.join(data_dir, image_name) for image_name
                                                             in stim['image_file_name'].values]))
    stim_set = StimulusSet(stim[stim.columns[:-1]])
    stim_set.stimulus_paths = image_paths # file path
    stim_set.identifier = stim_name # ID?

    return stim_set

def gen_BO_stim(BO_params, save_dir, stim_name):

    if not (os.path.isdir(save_dir)):
        os.mkdir(save_dir)

    if stim_name == 'dicarlo.Cirincione2022_border_ownership_standard_test':
        BO_stim = BO_Stimulus(save_dir=save_dir,
                          visual_degrees=BO_params[0],
                          stim_size=BO_params[1],
                          divisions=BO_params[2],
                          posx=BO_params[3],
                          posy=BO_params[4],
                          sqr_deg=BO_params[5])

    if stim_name == 'dicarlo.Cirincione2022_border_ownership_optimization_test':
        BO_optim = BO_Optim(save_dir=save_dir,
                        visual_degrees=BO_params[0],
                        stim_size=BO_params[1],
                        divisions=BO_params[2],
                        posx=BO_params[3],
                        posy=BO_params[4])


    return
