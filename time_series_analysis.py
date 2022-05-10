import numpy as np
import matplotlib.pyplot as plt
from scipy import stats as st
from scipy.signal import find_peaks
import seaborn as sns
import time
from statsmodels.tsa.stattools import grangercausalitytests
from sklearn.metrics import r2_score


class ts_analyser():
    def __init__(self, ts_data, epoch):
        #data
        self.epoch = epoch
        self.ts_data = ts_data
#         print('ts_analyser data shape: ', ts_data.shape)
        self.details()
    def details(self):
        #Distribution
        #plt.figure()
        self.g = sns.displot(data=self.ts_data, kind="kde", height=4, aspect=1).set(title='{} epochs'.format(self.epoch))
        

        
        pdf_y = self.g.ax.lines[0].get_ydata()
        pdf_x = self.g.ax.lines[0].get_xdata()
        pdf_a, _ = find_peaks(pdf_y, distance=50)
        
        self.deets = {
            'mean':np.mean(self.ts_data),
            'stdev':np.std(self.ts_data),
            'median':np.median(self.ts_data),
            'skew': st.skew(self.ts_data),
            'kurtosis':st.kurtosis(self.ts_data),
            #Long output
            'PDF':{
                #'x':pdf_x,
                #'y':pdf_y,
                'peaks':pdf_x[pdf_a]
            }
        }
    def comparison(self, data_new, title='', label=['original', 'new'], show=False):
#         print('comp shape: ',self.ts_data.shape, data_new.shape)
        #Kolmogorov-Smirnov test
        k_stat = st.ks_2samp(self.ts_data, data_new)
        self.KS = {
            'k_stat':k_stat[0],
            'k_p':k_stat[1]
        }
        #Granger Causality test
        #x = np.concatenate((np.expand_dims(self.ts_data, 1), np.expand_dims(data_new, 1)), axis=1), pvalue of 0.71
        x = np.concatenate((np.expand_dims(data_new, 1), np.expand_dims(self.ts_data, 1)), axis=1)
        g_stat = grangercausalitytests(x, maxlag=[1], verbose=0)
        self.granger = {'g_stat':g_stat[1][0]['ssr_ftest'][0],
                    'g_p': g_stat[1][0]['ssr_ftest'][1]}
        
        #Density distribution
        self.h = sns.displot(data={label[0]:self.ts_data, label[1]:data_new}, kind="kde", height=4, aspect=1).set(title='{}, {} epochs'.format(title, self.epoch), xlabel=str(self.KS))
        
        
        self.r_score = r2_score(data_new, self.ts_data)
        
        if show:
            self.h.savefig('comparison/KS_{}_dist_{}.png'.format(self.epoch, time.time()))
            self.g.savefig('distribution/{}_dist_{}.png'.format(self.epoch, time.time()))
            plt.show()
#         self.h.fig.clf() 
#         self.g.fig.clf() 
        plt.close('all') 
            