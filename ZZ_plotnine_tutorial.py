from plotnine import *
from plotnine.data import diamonds
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = diamonds



def pvscar(df):

    p = (
        ggplot(df)
         + aes(x='carat', y='price', color = 'cut') # what to plot
         + geom_point()                       # how to plot it the geometry
         )
    
    p.draw()
    p.save('pvscar.png')
    
    
    return None


def box_plot(df):
    p = (
        ggplot(df)
         + aes(x='cut', y='price') # what to plot
         + geom_boxplot()                       # how to plot it the geometry
         )
    
    p.draw()
    p.save('boxplot.png')
    
    return None


def hist(df):
    p = (
        ggplot(df)
         + aes(x='carat', fill = 'cut') # what to plot
         + geom_histogram(bins=20)                       # how to plot it the geometry
         )
    
    p.draw()
    p.save('pvscar.png')
    
    return None


dummy = pd.DataFrame({'x': np.random.normal(size=1000),
                      'y': np.random.normal(size=1000),
                      'group1': np.random.choice(['A', 'B', 'C'], 1000),
                        'group2': np.random.choice(['A', 'B', 'C'], 1000)})
def plot_reg(dummy):
    p1 = (
        ggplot(dummy)
        + aes(x='x', y='y')
        + geom_point() 
        + geom_smooth(method='lm')
    )

    p1.draw()
    p1.save('pvscar.png')

def facet(dummy):
    p = (
        ggplot(dummy)
            + aes(x='x', y='y', color='group1')
            + geom_point()
            + geom_smooth(method='lm')
            + facet_wrap('~group2')
            + theme_538()
            )
         
    p.draw()
    p.save('pvscar.png')

if __name__ == '__main__':
    facet(dummy)