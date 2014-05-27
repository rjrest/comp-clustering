import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
from pylab import *

def reportcard(id, df, df_norm):
    """ Plots a report for any competition id"""

    columns_array = np.empty(2,dtype=float)
    for indexed, elem in list(enumerate(df_norm.columns)):
        columns_array = np.vstack([columns_array, (indexed, elem) ])
    columns_array = columns_array[1:]
    
    # set up data
    bars_x = columns_array[:,0].astype(np.float)
    bars_axislabels = columns_array[:,1]
    bars_y = df_norm[df_norm.index == id].values[0]
    bars_title = 'Id=' + str(id)
    
    XX = np.array([1,2,3,8,10])
    
    Y2 = df[df.index == id].values[0,28:32]
    Y2 = np.append(Y2, 1)
    Y3 = df[df.index == id].values[0,32:36]
    Y3 = np.insert(Y3, 0, df[df.index == id].values[0,28])
    
    Y4 = df[df.index == id].values[0,46:50]
    Y4 = np.append(Y4, 1)
    Y5 = df[df.index == id].values[0,50:54]
    Y5 = np.insert(Y5, 0, df[df.index == id].values[0,46])
    
    Y6 = df[df.index == id].values[0,60:64]
    Y6 = np.append(Y6, 1)
    Y7 = df[df.index == id].values[0,64:68]
    Y7 = np.insert(Y7, 0, df[df.index == id].values[0,60])
    
    Y8 = df[df.index == id].values[0,36:41]
    Y9 = df[df.index == id].values[0,83:88]
    Y10 = df[df.index == id].values[0,93:98]
    Y11 = df[df.index == id].values[0,70]
    Y12 = df[df.index == id].values[0,69]
    Y13 = df[df.index == id].values[0,68]
    
    # layout
    fig = plt.figure(figsize=(12,16))
    G = gridspec.GridSpec(7, 3)
    
    # wide plot: full K-means behavior
    axes_1 = subplot(G[0:2, :], axisbg='0.96')
    axes_1.bar(bars_x, bars_y, width=0.8, facecolor='c', edgecolor='white')#, align='center')
    axes_1.set_title(bars_title, fontsize=18)
    _a1 = plt.xticks(bars_x, bars_axislabels.ravel(), rotation=90)
    
    # M9
    axes_2 = subplot(G[3,0], axisbg='0.94')
    axes_2.plot(XX, Y2)
    axes_2.set_title('M9. Forum activity %', fontsize=12, weight='bold')
    _a2_x = plt.xticks(XX, ['3','7','15','30','end'])
    _a2_y = plt.yticks(arange(0, 1.2, 0.2))
    # M9 Deltas
    axes_3 = subplot(G[4,0], axisbg='0.94')
    axes_3.bar(XX, Y3, width=0.5, facecolor='b',align='center')
    _a3_x = plt.xticks(XX, ['3','D7','D15','D30','end'])
    _a3_y = plt.yticks(arange(0, 1.2, 0.2))
    text(6, 0.9, 'deltas',ha='center',va='center',size=12,alpha=.33)
    
    # M13
    axes_4 = subplot(G[3,1], axisbg='0.96')
    axes_4.plot(XX, Y4)
    axes_4.set_title('M13. Subm activity %', fontsize=12, weight='bold')
    _a4_x = plt.xticks(XX, ['3','7','15','30','end'])
    _a4_y = plt.yticks(arange(0, 1.2, 0.2))
    # M13 Deltas
    axes_5 = subplot(G[4,1], axisbg='0.96')
    axes_5.bar(XX, Y5, width=0.5, facecolor='b',align='center')
    _a5_x = plt.xticks(XX, ['3','7','15','30','end'])
    _a5_y = plt.yticks(arange(0, 1.2, 0.2))
    text(6, 0.9, 'deltas',ha='center',va='center',size=12,alpha=.33)
    
    # M16
    axes_6 = subplot(G[3,2], axisbg='0.98')
    axes_6.plot(XX, Y6)
    axes_6.set_title('M16. User entry %', fontsize=12, weight='bold')
    _a6_x = plt.xticks(XX, ['3','7','15','30','end'])
    _a6_y = plt.yticks(arange(0, 1.2, 0.2))
    # M16 Deltas
    axes_7 = subplot(G[4,2], axisbg='0.98')
    axes_7.bar(XX, Y7, width=0.5, facecolor='b',align='center')
    _a7_x = plt.xticks(XX, ['3','7','15','30','end'])
    _a7_y = plt.yticks(arange(0, 1.2, 0.2))
    text(6, 0.9, 'deltas',ha='center',va='center',size=12,alpha=.33)
    
    # M10
    axes_8 = subplot(G[5,0], axisbg='0.94')
    axes_8.bar(XX, Y8, width=0.6, facecolor='y', edgecolor='white',align='center')
    #axes_8.set_title('M10. Forum word length', fontsize=12, weight='bold')
    _a8_x = plt.xticks(XX, ['3','7','15','30','end'])
    _a8_y = plt.yticks(arange(0,400,100))
    text(5, 250, 'M10. word length',ha='center',va='center',size=12,weight='bold')
    
    # M22
    axes_9 = subplot(G[5,1])
    axes_9.bar(XX, Y9, width=1, facecolor='c', edgecolor='white',align='center')
    #axes_9.set_title('M22. Relative all \npossible particip', fontsize=12, weight='bold')
    _a9_x = plt.xticks(XX, ['3','7','15','30','end'])
    _a9_y = plt.yticks(arange(0, 1.2, 0.2))
    text(6, 0.8, 'M22. % to all \npossible particip',ha='center',va='center',size=12,weight='bold')
    
    # M24
    axes_10 = subplot(G[5,2])
    axes_10.bar(XX, Y10, width=1, facecolor='0.75', edgecolor='white',align='center')
    #axes_10.set_title('M24. Sidelines rate', fontsize=12, weight='bold')
    _a10_x = plt.xticks(XX, ['3','7','15','30','end'])
    _a10_y = plt.yticks(arange(0, 1.2, 0.2))
    text(5.5, 0.30, 'M24.\nSidelines\nrate',ha='center',va='center',size=12)
    
    axes_11 = subplot(G[6,0], axisbg='0.94')
    xticks([]), yticks([])
    text(0.5, 0.7, 'M46. forum msgs\n/user/day',ha='center',va='center',size=12,weight='bold')
    text(0.5, 0.4, str(Y11)[0:5],ha='center',va='center',size=16)
    
    axes_12 = subplot(G[6,1:])
    xticks([]), yticks([])
    text(0.3, 0.7, 'M45. Subms/user/day:',ha='center',va='center',size=14)
    text(0.65, 0.7, str(Y12)[0:5],ha='center',va='center',size=14)
    text(0.3, 0.4, 'M44. anonymous users rate:',ha='center',va='center',size=14)  
    text(0.65, 0.4, str(format((Y13*100),".2f"))+'%',ha='center',va='center',size=14)
    #plt.savefig('../figures/gridspec.png', dpi=64)
    plt.show()
    return

# Helper function to make histograms
def make_histograms(df, suffix, fignum, fields, binns):
    """ Plots up to 8 columns using the methods 'raw', 'log10' or 'log', and with the number of bins as passed"""
    fig = plt.figure(num=fignum, figsize=(18,18))
    fig.suptitle('Histograms of ' + str(suffix) + ' features', fontsize=22)
    ax1 = fig.add_subplot(421, axisbg='0.94')
    ax2 = fig.add_subplot(422, axisbg='0.94')
    ax3 = fig.add_subplot(423, axisbg='0.94')
    ax4 = fig.add_subplot(424, axisbg='0.94')
    ax5 = fig.add_subplot(425, axisbg='0.94')
    ax6 = fig.add_subplot(426, axisbg='0.94')
    ax7 = fig.add_subplot(427, axisbg='0.94')
    ax8 = fig.add_subplot(428, axisbg='0.94')
    alphas = [0.33, 0.33, 0.6, 0.6, 0.28, 0.28, 0.6, 0.6]
    hues = ['g','b','b','g','g','b','b','g']
    all_axes = plt.gcf().axes
    for i, ax in list(enumerate(all_axes)):
        ax.set_ylabel("count", fontsize=10)
        for ticklabel in ax.get_xticklabels() + ax.get_yticklabels():
            ticklabel.set_fontsize(14)
        if (len(fields) - 1) >= i:
            if suffix == "raw":
                transformed = df[fields[i]].dropna().values
            elif suffix == "log10":
                transformed = np.log10(df[fields[i]].dropna().values)
            elif suffix == "log":
                transformed = np.log(df[fields[i]].dropna().values)
            
            #try:
            ax.hist(transformed, bins=binns[i], color=hues[i],alpha=alphas[i])
            ax.set_title(df[fields[i]].name, fontsize=20)
            #except:
            #    print "WARNING: An error occurred in composing {} Figure %d".format(str(suffix)) % fignum
            #    return
                       
    try:  # Save the figure as one file
        filename = "data/vis/histogram" + "_" + str(fignum) + "_" + str(suffix) + ".png"
        plt.savefig(filename)
        print "=  Vis Output: ", filename
    except IOError:
        print "WARNING: Failed to write out file: ", filename
        print
    plt.close(fig)

def compare_histograms(df, df_norm, fignum, fields, binns):
    """ Plots up to 4 columns as before and after normalization, using the number of bins as passed"""
    fig = plt.figure(num=fignum, figsize=(18,18))
    fig.suptitle('Histogram before and after normalization', fontsize=22)
    ax1 = fig.add_subplot(421, axisbg='0.94')
    ax2 = fig.add_subplot(422, axisbg='0.94')
    ax3 = fig.add_subplot(423, axisbg='0.94')
    ax4 = fig.add_subplot(424, axisbg='0.94')
    ax5 = fig.add_subplot(425, axisbg='0.94')
    ax6 = fig.add_subplot(426, axisbg='0.94')
    ax7 = fig.add_subplot(427, axisbg='0.94')
    ax8 = fig.add_subplot(428, axisbg='0.94')
    alphas = [0.33, 0.33, 0.6, 0.6, 0.28, 0.28, 0.6, 0.6]
    hues = ['g','y','g','y','g','y','g','y']
    all_axes = plt.gcf().axes
    # print list(enumerate(fields))
    for i, ax in list(enumerate(all_axes)):
        ax.set_ylabel("count", fontsize=10)
        for ticklabel in ax.get_xticklabels() + ax.get_yticklabels():
            ticklabel.set_fontsize(14)
        g = np.int(math.ceil(np.float(i)/2))
        
        if (len(fields)*2-1) >= i:
            if i in (0,2,4,6):
                ax.hist(df[fields[i-g]].dropna().values, bins=binns[i-g], color=hues[i],alpha=alphas[i])
                print "  plot " + str(df[fields[i-g]].name)
                ax.set_title(df[fields[i-g]].name, fontsize=20)
        #if (len(fields)*2) >= i:   
            if i in (1,3,5,7):
                #try:
                ax.hist(df_norm[fields[i-g]].dropna().values, bins=binns[i-g], color=hues[i],alpha=alphas[i])
                ax.set_title("As normalized:", fontsize=20)
                       
    try:  # Save the figure as one file
        filename = "data/vis/histogram_compare" + "_" + str(fignum) + ".png"
        plt.savefig(filename)
        print "=  Vis Output: ", filename
        print
    except IOError:
        print "WARNING: Failed to write out file: ", filename
        print
    plt.close(fig)