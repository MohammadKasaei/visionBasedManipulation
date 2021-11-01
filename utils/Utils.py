from math import acos, asin, atan2, cos, pi, sin
import numpy as np
import sys

try:
    GLOBAL_DIR = sys._MEIPASS #temporary folder with libs & data files
except:
    GLOBAL_DIR = "."

class Utils():
    '''
    This class provides:
        - general mathematical operations that are not directly available through numpy 
        - file management utilities (since files may be located in temp folder during runtime)
    '''

    @staticmethod
    def deg_sph2cart(spherical_vec):
        ''' Converts SimSpark's spherical coordinates in degrees to cartesian coordinates '''
        r = spherical_vec[0]
        h = spherical_vec[1] * pi / 180
        v = spherical_vec[2] * pi / 180
        return np.array([r * cos(v) * cos(h), r * cos(v) * sin(h), r * sin(v)])

    @staticmethod
    def deg_sin(deg_angle):
        ''' Returns sin of degrees '''
        return sin(deg_angle * pi / 180)

    @staticmethod
    def deg_cos(deg_angle):
        ''' Returns cos of degrees '''
        return cos(deg_angle * pi / 180)

    @staticmethod
    def to_3d(vec_2d, value=0) -> np.ndarray:
        ''' Returns new 3d vector from 2d vector '''
        return np.append(vec_2d,value)

    @staticmethod
    def to_2d_as_3d(vec_3d) -> np.ndarray:
        ''' Returns new 3d vector where the 3rd dimension is zero '''
        vec_2d_as_3d = np.copy(vec_3d)
        vec_2d_as_3d[2] = 0
        return vec_2d_as_3d

    @staticmethod
    def get_active_directory(dir:str) -> str:
        global GLOBAL_DIR
        return GLOBAL_DIR + dir

    @staticmethod
    def acos(val):
        ''' arccosine function that limits input '''
        return acos( np.clip(val,-1,1) )
    
    @staticmethod
    def asin(val):
        ''' arcsine function that limits input '''
        return asin( np.clip(val,-1,1) )

    @staticmethod
    def normalize_deg(val):
        ''' normalize val in range [-180,180[ '''
        return (val + 180.0) % 360 - 180

    @staticmethod
    def normalize_rad(val):
        ''' normalize val in range [-pi,pi[ '''
        return (val + pi) % (2*pi) - pi

    @staticmethod
    def get_target_angle_deg(pos2d, target):
        ''' get angle (degrees) of vector (target-pos2d) '''
        return atan2(target[1]-pos2d[1], target[0]-pos2d[0]) * 180 / pi

    @staticmethod
    def get_target_angle_deg_2(pos2d, ori, target):
        ''' get angle (degrees) of target if we're located at 'pos2d' with orientation 'ori' (degrees) '''
        return Utils.normalize_deg( atan2(target[1]-pos2d[1], target[0]-pos2d[0]) * 180 / pi - ori )

    @staticmethod
    def get_point_distance_to_line_2d(l1,l2,p):
        ''' 
        Get distance between point p and line (l1,l2), and side where p is

        Parameters
        ----------
        l1 : ndarray
            2D point that defines line
        l2 : ndarray
            2D point that defines line
        p : ndarray
            2D point

        Returns
        -------
        distance : float
            distance between line and point
        side : str
            if we are at l1, looking at l2, p may be at our "left" or "right"
        '''
        line_len = np.linalg.norm(l2-l1)

        if line_len == 0: # assumes vertical line
            dist = sdist = np.linalg.norm(p-l1)
        else:
            sdist = np.cross(l2-l1,p-l1)/line_len
            dist = abs(sdist)

        return dist, "left" if sdist>0 else "right"
        


