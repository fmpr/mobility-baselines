# @Time     : Feb. 11, 2022 15:15
# @Author   : Filipe Rodrigues
# @FileName : mobility_detrender.py

import datetime
import numpy as np
from matplotlib import pyplot as plt
from evaluation import evaluation

class MobilityDetrender():
    
    def __init__(self, day_duration=12*24, week_duration=7):
        self.day_duration = day_duration
        self.week_duration = week_duration

    def fit(self, data, start_date, holidays=[]):
        # compute historical averages
        sums = np.zeros((data.shape[1], self.week_duration, self.day_duration))
        sums_sq = np.zeros((data.shape[1], self.week_duration, self.day_duration))
        self.counts = np.zeros((data.shape[1], self.week_duration, self.day_duration))
        offset = start_date.weekday()
        for i in range(len(data)):
            day = int(i / (self.day_duration))
            weekday = int((day+offset) % self.week_duration) # adjust with +offset to get the right weekday assignment
            if day in holidays:
                weekday = 6 # treat holidays as Sundays (weekday=6)
            tod = int(i % (self.day_duration))
            sums[:,weekday,tod] += data[i]
            sums_sq[:,weekday,tod] += (data[i]**2)
            self.counts[:,weekday,tod] += 1.0

        self.historical_avg = sums / self.counts
        self.historical_std = np.sqrt((sums_sq / self.counts) - (self.historical_avg**2) + 0.1)
        
    def visualize(self, loc):
        fig, axs = plt.subplots(3, figsize=(16,6))
        img = axs[0].imshow(self.counts[loc,:,:])
        axs[0].set_title("Observations counts")
        axs[0].set_xlabel("Time of day")
        axs[0].set_ylabel("Weekday")
        plt.colorbar(img, ax=axs[0])
        img = axs[1].imshow(self.historical_avg[loc,:,:])
        axs[1].set_title("Historical average")
        axs[1].set_xlabel("Time of day")
        axs[1].set_ylabel("Weekday")
        plt.colorbar(img, ax=axs[1])
        img = axs[2].imshow(self.historical_std[loc,:,:])
        axs[2].set_title("Historical standard dev.")
        axs[2].set_xlabel("Time of day")
        axs[2].set_ylabel("Weekday")
        plt.colorbar(img, ax=axs[2])
        plt.show()
        
    def transform(self, data, start_date, holidays=[], mode='avg'):
        # compute predictions based on historical averages for entire dataset
        trues = []
        ha_trend = []
        ha_trend_std = []
        offset = start_date.weekday()
        for i in range(len(data)):
            trues.append(data[i])
            day = int(i / (self.day_duration))
            weekday = int((day+offset) % self.week_duration) # adjust with +offset to get the right weekday assignment
            if day in holidays:
                weekday = 6 # treat holidays as Sundays (weekday=6)
            tod = int(i % (self.day_duration))
            ha_trend.append(self.historical_avg[:,weekday,tod])
            ha_trend_std.append(self.historical_std[:,weekday,tod])

        self.ha_trend = np.array(ha_trend)
        self.ha_trend_std = np.array(ha_trend_std)
        self.trues = np.array(trues)
        
        # standardize data according to historical averages
        if mode == 'avg':
            detrended_data = self.trues - self.ha_trend
        elif mode == 'avg+std':
            detrended_data = (self.trues - self.ha_trend) / self.ha_trend_std
        else:
            raise Exception('Unknown mode:', mode)
        
        return detrended_data