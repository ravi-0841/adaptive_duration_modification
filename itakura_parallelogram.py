#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Code for Dynamic Time Warping and its variants.

# Author: Johann Faouzi <johann.faouzi@gmail.com>
# License: BSD-3-Clause

"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import check_array

#%%
class ItakuraParallelogram(object):
    
    def __init__(self, slope=1.0):
        self.max_slope = slope
        self.region = None
        self.upper_bound = 2**32 - 1


    def _check_region(self, region, n_timestamps_1, n_timestamps_2):
        """Project region on the feasible set."""
        region = np.clip(region[:, :n_timestamps_1], 0, n_timestamps_2)
        return region
    
    
    def _get_itakura_slopes(self, n_timestamps_1, n_timestamps_2, max_slope):
        """Compute the slopes of the parallelogram bounds."""
        if not isinstance(n_timestamps_1, (int, np.integer)):
            raise TypeError("'n_timestamps_1' must be an integer.")
        else:
            if not n_timestamps_1 >= 2:
                raise ValueError("'n_timestamps_1' must be an integer greater than"
                                 " or equal to 2.")
    
        if not isinstance(max_slope, (int, np.integer, float, np.floating)):
            raise TypeError("'max_slope' must be an integer or a float.")
        else:
            if not max_slope >= 1:
                raise ValueError("'max_slope' must be a number greater "
                                 "than or equal to 1.")
    
        min_slope = 1 / max_slope
        scale_max = (n_timestamps_2 - 1) / (n_timestamps_1 - 2)
        max_slope *= scale_max
        max_slope = max(1., max_slope)
    
        scale_min = (n_timestamps_2 - 2) / (n_timestamps_1 - 1)
    
        min_slope *= scale_min
        min_slope = min(1., min_slope)
        return min_slope, max_slope


    def _accumulated_cost_matrix_region(self, cost_matrix, region):
            n_timestamps_1, n_timestamps_2 = cost_matrix.shape
            acc_cost_mat = np.ones((n_timestamps_1, n_timestamps_2)) * np.inf
            acc_cost_mat[0, 0: region[1, 0]] = np.cumsum(
                cost_matrix[0, 0: region[1, 0]]
            )
            acc_cost_mat[0: region[1, 0], 0] = np.cumsum(
                cost_matrix[0: region[1, 0], 0]
            )
            region_ = np.copy(region)
            region_[0] = np.maximum(region_[0], 1)
            for i in range(1, n_timestamps_1):
                for j in range(region_[0, i], region_[1, i]):
                    acc_cost_mat[i, j] = cost_matrix[i, j] + min(
                        acc_cost_mat[i - 1][j - 1],
                        acc_cost_mat[i - 1][j],
                        acc_cost_mat[i][j - 1]
                    )
            return acc_cost_mat


    def return_path(self, acc_cost_mat):
        acc_cost_mat = acc_cost_mat.T
        n_timestamps_1, n_timestamps_2 = acc_cost_mat.shape
        path = [(n_timestamps_1 - 1, n_timestamps_2 - 1)]
        while path[-1] != (0, 0):
            i, j = path[-1]
            if i == 0:
                path.append((0, j - 1))
            elif j == 0:
                path.append((i - 1, 0))
            else:
                arr = np.array([acc_cost_mat[i - 1][j - 1],
                                acc_cost_mat[i - 1][j],
                                acc_cost_mat[i][j - 1]])
                argmin = np.argmin(arr)
                if argmin == 0:
                    path.append((i - 1, j - 1))
                elif argmin == 1:
                    path.append((i - 1, j))
                else:
                    path.append((i, j - 1))
        return np.transpose(np.array(path)[::-1])


    def return_constrained_path(self, acc_cost_mat, steps_limit=2):
        acc_cost_mat = acc_cost_mat.T
        n_timestamps_1, n_timestamps_2 = acc_cost_mat.shape
        vertical_moves = 0
        horizontal_moves = 0
        path = [(n_timestamps_1 - 1, n_timestamps_2 - 1)]
        while path[-1] != (0, 0):
            i, j = path[-1]
            if i == 0:
                path.append((0, j-1))
            elif j == 0:
                path.append((i-1, 0))
            else:
                if vertical_moves >= steps_limit and horizontal_moves >= steps_limit:
                    arr = np.array([acc_cost_mat[i-1,j-1], 
                                    self.upper_bound, 
                                    self.upper_bound])
                    vertical_moves = 0
                    horizontal_moves = 0
                    argmin = np.argmin(arr)

                if vertical_moves >= steps_limit and horizontal_moves < steps_limit:
                    arr = np.array([acc_cost_mat[i-1,j-1],
                                    self.upper_bound, 
                                    acc_cost_mat[i,j-1]])
                    vertical_moves = 0
                    argmin = np.argmin(arr)
                
                elif horizontal_moves >= steps_limit and vertical_moves < steps_limit:
                    arr = np.array([acc_cost_mat[i-1,j-1], 
                                    acc_cost_mat[i-1,j], 
                                    self.upper_bound])                    
                    horizontal_moves = 0
                    argmin = np.argmin(arr)

                else:
                    arr = np.array([acc_cost_mat[i-1,j-1], 
                                    acc_cost_mat[i-1,j], 
                                    acc_cost_mat[i,j-1]])
                    argmin = np.argmin(arr)
                
                if argmin == 0:
                    path.append((i-1, j-1))
                    vertical_moves = 0
                    horizontal_moves = 0
                elif argmin == 1:
                    path.append((i-1, j))
                    vertical_moves += 1
                    horizontal_moves = 0
                else:
                    path.append((i, j-1))
                    horizontal_moves += 1
                    vertical_moves = 0
        return np.transpose(np.array(path)[::-1])


    def itakura_parallelogram(self, n_timestamps_1, n_timestamps_2=None, max_slope=2.):
        """Compute the Itakura parallelogram.
        Parameters
        ----------
        n_timestamps_1 : int
            The size of the first time series. (goes in columns)
        n_timestamps_2 : int (optional, default None)
            The size of the second time series. If None, set to `n_timestamps_1`. (goes in rows)
        max_slope : float (default = 2.)
            Maximum slope for the parallelogram. Must be >= 1.
        Returns
        -------
        region : array, shape = (2, n_timestamps_1)
            Constraint region. The first row consists of the starting indices
            (included) and the second row consists of the ending indices (excluded)
            of the valid rows for each column. 
        References
        ----------
        .. [1] F. Itakura, "Minimum prediction residual principle applied to
               speech recognition". IEEE Transactions on Acoustics,
               Speech, and Signal Processing, 23(1), 67–72 (1975).
        Examples
        --------
        >>> from pyts.metrics import itakura_parallelogram
        >>> print(itakura_parallelogram(5))
        [[0 1 1 2 4]
         [1 3 4 4 5]]
        """
        if n_timestamps_2 is None:
            n_timestamps_2 = n_timestamps_1
        min_slope_, max_slope_ = self._get_itakura_slopes(
            n_timestamps_1, n_timestamps_2, max_slope)
    
        # Now we create the piecewise linear functions defining the parallelogram
        # lower_bound[0] = min_slope * x
        # lower_bound[1] = max_slope * (x - n_timestamps_1) + n_timestamps_2
    
        centered_scale = np.arange(n_timestamps_1) - n_timestamps_1 + 1
        lower_bound = np.empty((2, n_timestamps_1))
        lower_bound[0] = min_slope_ * np.arange(n_timestamps_1)
        lower_bound[1] = max_slope_ * centered_scale + n_timestamps_2 - 1
    
        # take the max of the lower linear funcs
        lower_bound = np.round(lower_bound, 2)
        lower_bound = np.ceil(np.max(lower_bound, axis=0))
    
        # upper_bound[0] = max_slope * x
        # upper_bound[1] = min_slope * (x - n_timestamps_1) + n_timestamps_2
    
        upper_bound = np.empty((2, n_timestamps_1))
        upper_bound[0] = max_slope_ * np.arange(n_timestamps_1) + 1
        upper_bound[1] = min_slope_ * centered_scale + n_timestamps_2
    
        # take the min of the upper linear funcs
        upper_bound = np.round(upper_bound, 2)
        upper_bound = np.floor(np.min(upper_bound, axis=0))
    
        # Little fix for max_slope = 1
        if max_slope == 1:
            if n_timestamps_2 > n_timestamps_1:
                upper_bound[:-1] = lower_bound[1:]
            else:
                upper_bound = lower_bound + 1
    
        region = np.asarray([lower_bound, upper_bound]).astype('int64')
        region = self._check_region(region, n_timestamps_1, n_timestamps_2)
        self.region = region
        return region
    

    def accumulated_cost_matrix(self, cost_mat, region=None):
        """Compute the accumulated cost matrix.
        Parameters
        ----------
        cost_mat : array-like, shape = (n_timestamps_2, n_timestamps_1)
            Cost matrix.
        region : None or tuple, shape = (2, n_timestamps_1) (default = None)
            Constraint region. If None, there is no constraint region.
            If array-like, the first row indicates the starting indices (included)
            and the second row the ending indices (excluded) of the valid rows
            for each column.
        Returns
        -------
        acc_cost_mat : array, shape = (n_timestamps_2, n_timestamps_1)
            Accumulated cost matrix.
        """
        cost_mat = cost_mat.T
        cost_mat = check_array(cost_mat, ensure_min_samples=2,
                               ensure_min_features=2, ensure_2d=True,
                               force_all_finite=False, dtype='float')
        cost_mat_shape = cost_mat.shape
    
        if region is None:
            region = self.region
        else:
            region = check_array(region, dtype='int64')

        region_shape = region.shape
        if region_shape != (2, cost_mat_shape[0]):
            raise ValueError("The shape of 'region' must be equal to "
                             "(2, n_timestamps_1) "
                             "(got {0})".format(region_shape)
                             )
        acc_cost_mat = self._accumulated_cost_matrix_region(cost_mat, region)
        return acc_cost_mat.T
    
    
    def itakura_mask(self, n_timestamps_1, n_timestamps_2=None, max_slope=2.0):
        
        if n_timestamps_2 is None:
            n_timestamps_2 = n_timestamps_1
        
        cords = self.itakura_parallelogram(n_timestamps_1, n_timestamps_2, max_slope)
        
        mask = np.zeros((n_timestamps_2, n_timestamps_1))
        for col in range(n_timestamps_1):
            mask[cords[0,col]:cords[1,col],col] = 1.
        
        return mask
    
    
    def plot_itakura(self, n_timestamps_1, n_timestamps_2, max_slope=1., ax=None):
        """Plot Itakura parallelogram."""
        region = self.itakura_parallelogram(n_timestamps_1, n_timestamps_2, max_slope)
        max_slope, min_slope = self._get_itakura_slopes(
            n_timestamps_1, n_timestamps_2, max_slope)
        mask = np.zeros((n_timestamps_2, n_timestamps_1))
        for i, (j, k) in enumerate(region.T):
            mask[j:k, i] = 1.
    
        plt.imshow(mask, origin='lower', cmap='Wistia')
    
        sz = max(n_timestamps_1, n_timestamps_2)
        x = np.arange(-1, sz + 1)
    
        low_max_line = ((n_timestamps_2 - 1) - max_slope * (n_timestamps_1 - 1)) +\
            max_slope * np.arange(-1, sz + 1)
        up_min_line = ((n_timestamps_2 - 1) - min_slope * (n_timestamps_1 - 1)) +\
            min_slope * np.arange(-1, sz + 1)
        diag = (n_timestamps_2 - 1) / (n_timestamps_1 - 1) * np.arange(-1, sz + 1)
        plt.plot(x, diag, 'black', lw=1)
        plt.plot(x, max_slope * np.arange(-1, sz + 1), 'b', lw=1.5)
        plt.plot(x, min_slope * np.arange(-1, sz + 1), 'r', lw=1.5)
        plt.plot(x, low_max_line, 'g', lw=1.5)
        plt.plot(x, up_min_line, 'y', lw=1.5)
    
        for i in range(n_timestamps_1):
            for j in range(n_timestamps_2):
                plt.plot(i, j, 'o', color='green', ms=1)
    
        ax.set_xticks(np.arange(-.5, n_timestamps_1, 1), minor=True)
        ax.set_yticks(np.arange(-.5, n_timestamps_2, 1), minor=True)
        plt.grid(which='minor', color='b', linestyle='--', linewidth=1)
        plt.xticks(np.arange(0, n_timestamps_1, 2))
        plt.yticks(np.arange(0, n_timestamps_2, 2))
        plt.xlim((-0.5, n_timestamps_1 - 0.5))
        plt.ylim((-0.5, n_timestamps_2 - 0.5))

#%%
if __name__=="__main__":

    slopes = [1., 1.5, 3.]
    rc = {"font.size": 14, "axes.titlesize": 10,
          "xtick.labelsize": 8, "ytick.labelsize": 8}
    plt.rcParams.update(rc)
    
    
    lengths = [(10, 10), (10, 5), (5, 10)]
    y_coordinates = [0.915, 0.60, 0.35]
    
    plt.figure(figsize=(10, 8))
    
    itakura_object = ItakuraParallelogram()
    
    for i, ((n1, n2), y) in enumerate(zip(lengths, y_coordinates)):
        for j, slope in enumerate(slopes):
            ax = plt.subplot(3, 3, i * 3 + j + 1)
            itakura_object.plot_itakura(n1, n2, max_slope=slope, ax=ax)
            plt.title('max_slope = {}'.format(slope))
            if j == 1:
                plt.figtext(0.5, y, 'itakura_parallelogram({}, {})'.format(n1, n2),
                            ha='center')
    plt.subplots_adjust(hspace=0.4)
    plt.show()