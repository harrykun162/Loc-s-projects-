import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plt_species(x, y, species):
    """ Scatter plot of the iris species, with legend."""
    # categories and iris have already been defined, so are picked up by the function
    cats = species.unique()
    for cat in cats: # create a vector with the category names,
        # go through all the entries in categories
        # scatter plot in 2d of these species which match cat, and give them size=10 and label cat
        plt.scatter(x[species == cat], y[species == cat], s=10, label=cat)
    # plot a legend with the labels in the best location on the figure
    plt.legend(loc='best')

def plt_confusion_matrix(cnf_matrix, cats, method):
    """
    Plots a sklearn confusion matrix with categories 'cats' for a classifier 'method'
    """
    # write the confusion matrix to a dataframe with row and column names as the categories, which are already defined
    cmatrix = pd.DataFrame(cnf_matrix,columns=cats,index=cats) 
    f, ax = plt.subplots(figsize=(7,6)) # initialise the plots and axes
    sns.heatmap(cmatrix, annot=True, linewidths=.5) # plot the confusion matrix as a heatmap
    plt.title('Confusion matrix for '+method) # add a title, + concatenates two strings
    plt.ylabel('Actual label') # add a ylabel
    plt.xlabel('Predicted label') # add a xlabel
    # adjust the bottom and top of the figure, so we can view all of it
    bottom, top = ax.get_ylim()  # get the y axis limits
    ax.set_ylim(bottom + 0.5, top - 0.5); # adjust the y axis limits

def plt_decision_boundaries(skm, xx, yy, ax=None, fill=True):
    """
    Takes a sklearn model (skm) with two features xx and yy and plots the decision boundaries.
    """
    ax = ax or plt.gca()
    # ravel is a numpy method which converts a two-dimensional array of size (n,m) to a vector of length nm
    # column_stack is a numpy function which takes two column arrays of length N 
    # and creates a two-dimensional array of size (N,2)
    # now pass the (N,2) array to the model and predict values based on these features, zz will have size (N,1)
    zz = skm.predict(np.column_stack([xx.ravel(), yy.ravel()]))  
    zz = zz.reshape(xx.shape) # reshape zz so it has the size of the original array xx, i.e., (n,m)
    if fill:
        ax.contourf(xx, yy, zz, cmap=plt.cm.Paired) # plot the decision boundaries as filled contours
    else:
        ax.contour(xx, yy, zz, linewidths=.25) # plot the decision boundaries as filled contours
    
def plt_correlation_matrix(corrs):
    '''Uses the seaborn heatmap to plot the correlation matrix of a pandas dataframe''' 
    # as this is a symmetric table, set up a mask so that we only plot values 
    # below the main diagonal
    mask = np.triu(np.ones_like(corrs, dtype=np.bool))
    f, ax = plt.subplots(figsize=(10, 8)) # initialise the plots and axes
    # plot the correlations as a seaborn heatmap, with a colourbar
    sns.heatmap(corrs, mask=mask, center=0, annot=True, square=True, linewidths=.5) 
    # do some fiddling so that the top and bottom are not obscured
    bottom, top = ax.get_ylim() 
    ax.set_ylim(bottom + 0.5, top - 0.5)

