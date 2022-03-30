# -*- coding: utf-8 -*-
"""
@author: Anselm Ng

A collection of functions used in Model.ipynb
"""

import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib import gridspec
import pandas as pd


#%% Extracting input data and appending it into a single predictor and target variable
def extractData(dataDir, setting, settingRatio = None, totalImages = 86, combinedHotspot = False):
    
    '''
    Function to extract and append input image (fire and/or lava images) into a NumPy arrays for the predictor variables (X)
    and the target variables (y), along with associated feature (column) names of X.
    
    Inputs:
        dataDir --> Directory of the folder containing lava and fire image subfolders (In which each subfolder contains all
                    the sub-subfolders for each image (with .tif and .shp files).
        setting --> Setting for the algorithm to be trained on in terms of input image type. Must be defined as a list of
                    tuples, with each tuple containing satellite followed by hotspot name. (i.e., if we want out input dataset 
                    to have only S2 fire and LS8 lava images, we define setting as [('S2', 'fire'), ('LS8', 'lava')] ).
        settingRatio --> Ratio for each tuple in setting with respect to totalImages. Must be defined when there is more
                         than 1 component in setting. Defined as a list of integers for each setting in sequence (i.e., if
                         we want our setting to be 1-S2F and 2-LS8, we define settingRatio as [1,2]). Defaults to None.
        totalImages --> Total number of input image to consider when building the predictor and target variables (X & y).
                        Defaults to 86.
        combinedHotspot --> Boolean variable that decides if lava and fire labels in y should all be converted to a single
                            label (hotspot) if set to True. Defauts to False.
        
    Outputs:
        X --> Predictor variable matrix of size (all labelled pixels, 10).
        y --> Target variable vector of size (all labelled pixels,)
        features --> NumPy array of feature names corresponding to each of the 10 columns of X.
        
    '''
    
    import rasterio, rasterio.mask
    import geopandas as gpd
    
    # Defining all feature names.
    features = ['Ultra-Blue','Blue','Green','Red','NIR','SWIR1','SWIR2','NDVI','NBR','IWCD']
    
    if len(setting) > 1 and len(settingRatio) != len(setting):
        raise ValueError('Length of settingRatio argument does not match length of setting argument.')
        

    # Cycling through each tuple (component) in setting.
    for satelliteIdx, Tuple in enumerate(setting):

        # Defining the satellite and the corresponding hotspot class of the data/image.
        satellite = Tuple[0].upper()
        hotspot = Tuple[1].lower()

        # Defining the directory that holds the current component's dataset (parentDir), and all its sub-directories (caseList).
        parentDir = os.path.join(dataDir,hotspot.capitalize() + ' Images',satellite)
        caseList = os.listdir(parentDir)

        # If the given setting is a composite of different image input types, the number of input images will have to be
        # scaled based on totalImages.
        if len(setting) > 1:
            
            

            # Obtaining a weightage of input images before sampling them.
            groupList = list(map(lambda i: i.split('-')[0], caseList)) # All unique groups (volcanoes/grid for fire) in caseList
            groupProb = 1/len(set(groupList)) # (assumed equal) Probability of each group's occurrence
            weightedProb = [groupProb/(groupList.count(group)) for group in groupList] # Weight it by their frequency

            # Sample (without replacement) caseList based on the volcano's weighted probabilities.
            portionLen = np.ceil(totalImages * (settingRatio[satelliteIdx]/sum(settingRatio))).astype(int)
            if portionLen > len(caseList):
                raise ValueError("There isn't enough input images for %s-%s (%d current) for this setting ratio (%d needed) with respect to totalImages (%d total)." % (satellite,hotspot,len(caseList),portionLen,totalImages))
            caseList = np.random.choice(caseList, size=portionLen, replace=False, p=weightedProb) # Sampling


        # Now extracting relevant raster (.tif) and vector/shape (.shp) data for each input image folder in caseList.
        for childDirIdx, childDirName in enumerate(caseList):

            # Defining respective raster and vector file names.
            childDir = os.path.join(parentDir,childDirName)
            shapefileIn = os.path.join(childDir,'vector.shp')
            rasterfileIn = os.path.join(childDir,'raster.tif')

            # Reading the entire shape file into a GeoPandas dataframe.
            # Keeping the rasterfile open in order to extract raster values within the shapefile's polygon geometries.
            shapefile = gpd.read_file(shapefileIn)
            with rasterio.open(rasterfileIn,'r') as rasterfile:

                # If both coordinate systems are unequal, change the shape file crs to match the raster file's crs.
                # First extract the raster file's crs and convert the shape file's crs thereafter.
                if rasterfile.crs != shapefile.crs:
                    rasterCRS = rasterfile.crs.to_dict()['init']
                    shapefile = shapefile.to_crs(rasterCRS)

                # Number of bands (features) to consider.
                nBands = rasterfile.count

                # X (features / predictor variables) and y (target variables).
                if childDirIdx == 0 and satelliteIdx == 0:
                    X = np.array([], dtype = np.float16).reshape(0,nBands) # Feature array is 0 x nBands long (prevent error when appending)
                    y = np.array([], dtype='str') # Specific for string labelled target variables (i.e., fire, cloud, etc).

                # Getting all the polygon values from the geometry column 
                geometryList = shapefile['geometry'].values

                # Looping through each polygon (geometry) to extract the corresponding pixel values from the raster dataset
                for idx, geometry in enumerate(geometryList):

                    # Converting the polygon into a GeoJSON formatted dictionary
                    featurePolygon = [geometry.__geo_interface__]

                    # Obtaining only the pixels in the raster dataset that corresponds to the polygon
                    out_image, out_transform = rasterio.mask.mask(rasterfile,featurePolygon,crop=True)

                    # Taking out all the 0 values created from the mask
                    # Then forcing the matrix into a nx3 array (3 columns for 3 bands)
                    # Each row here therefore represents a pixel
                    out_image_reshaped = out_image[:,~np.all(out_image==0,axis=0)].T

                    # Labelling every pixel (y)
                    # Appending the corresponding pixel row (X)
                    y = np.append(y, [shapefile['class'][idx]] * out_image_reshaped.shape[0])
                    X = np.vstack((X,out_image_reshaped))

    # Converting features into a np array.
    features = np.array(features)
    
    # Converting all fire and lava pixels to a single target hotspot class.
    if combinedHotspot:
        y = np.char.replace(y, 'lava', 'hotspot')
        y = np.char.replace(y, 'fire', 'hotspot')
    
    return (X, y, features)




#%% Computing single-class training error in multi-label classification for random forest models
def getTrainingError(model, X_train, y_train, Class, k = 10, random_state = 1, 
                     verbosity = False, compareTestErr = None):
    
    '''
    Computes training error for a random forest for precision, recall and F1-score for a single class. By design, 
    random forest training errors are typically very close to 0 based on certain settings (i.e. min_samples_leaf = 1),
    where it is hard to estimate overfitting when compared to test errors. Therefore, an alternative to computing this
    training error is to train a k-fold cross validation on the training set itself, and extract the average error of
    the computed results.
    
    It is important to note that this function computes purely the precision, recall & F1-score errors of a single class
    in a multi-label classification scenario. Errors taking into account all class (i.e., weighted average) can be obtained
    by running the standard `cross_val_score` function that sklearn provides.
    
    Following the suggestion from:
        # https://stats.stackexchange.com/questions/162353/what-measure-of-training-error-to-report-for-random-forests
    
    
    Inputs:
        model -> Base random forest model with pre-defined parameters.
        X_train -> Matrix of training input features (n_datapoints, m_features).
        y_train -> Vector of training labels (n_datapoints,).
        Class -> Specific class for which error is to be computed on.
        k -> Number of cross validation sets [Defaults at 10].
        random_state -> For consistency [Defaults at 1].
        verbosity -> Prints the errors for the 3 metrics along with their SDs to the screen if True [Defaults to False].
        compareTestErr -> The same format as cvMeanScores, just that it represents the test errors instead of the training
                          errors computed in this function. Used to compare (visually) the test error alongside the training
                          error ONLY when verbosity is set to True [Defaults to None].
        
    Outputs:
        cvMeanScores -> A 3-element NumPy array consisting of the mean error for `Class` for Precision, Recall & F1-score
                        respectively in that order.
    '''

    # Only need to import this function when we run it
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import precision_recall_fscore_support

    # Obtaining the indices of the split
    cvSplit = StratifiedKFold(k, shuffle=True, random_state=random_state).split(X_train, y_train)

    # Initializing vectors to hold the results of each cross validation set
    cvScoreVect = {'Precision': np.zeros(k), 'Recall': np.zeros(k),'fScore': np.zeros(k)}
    cvCount = 0 # For enumerating through each cross validation set
    

    for train_splitIdx, test_splitIdx in cvSplit:

        # Defining the cross validation set's training data
        X_split_train = X_train[train_splitIdx, :]
        y_split_train = y_train[train_splitIdx]

        # Defining the cross validation set's testing data
        X_split_test = X_train[test_splitIdx, :]
        y_split_test = y_train[test_splitIdx]

        # Training the model and predicting the test data
        model.fit(X_split_train, y_split_train)
        splitPred = model.predict(X_split_test)
        
        # Defining available classes, and obtaining the index of `Class`
        splitClasses = np.unique(y_split_train)
        hotspotClass = splitClasses.tolist().index(Class)

        # Obtaining the metric scores for the predicted data
        splitScores = precision_recall_fscore_support(y_split_test,splitPred,average=None,beta=1,labels=np.unique(y_split_train))

        # Appending the metric scores for the predicted data
        cvScoreVect['Precision'][cvCount] = splitScores[0][hotspotClass]
        cvScoreVect['Recall'][cvCount] = splitScores[1][hotspotClass]
        cvScoreVect['fScore'][cvCount] = splitScores[2][hotspotClass]

        cvCount += 1 # Enumerator

    # Obtaining the error for each metric as a NumPy array
    cvMeanScores = 1 - np.array([np.nanmean(cvScoreVect['Precision']), np.nanmean(cvScoreVect['Recall']), np.nanmean(cvScoreVect['fScore'])])

    
    ## Just for checking purposes (if verbosity is true)
    if verbosity:
        # Computing SDs for each metric's training error.
        cvStdScores = [np.nanstd(cvScoreVect['Precision']), np.nanstd(cvScoreVect['Recall']), np.nanstd(cvScoreVect['fScore'])]
        
        # Creating a boolean for test error comparison.
        if len(compareTestErr) == len(cvMeanScores):
            compareTest = True
        else:
            compareTest = False
            
        # Printing the respective results for each metric thereafter.
        print('Arbitrary Training Error for class %s (%d-fold CV of training dataset) | Computed Test Error' % (Class, k))
        for cvIdx, cvMetric in enumerate(cvScoreVect.keys()):
            if compareTest: # Comparing to test error
                print('%s Error: (Training) %.3g with std of %.3g | (Testing) %.3g'\
                      % (cvMetric, cvMeanScores[cvIdx], cvStdScores[cvIdx], compareTestErr[cvIdx]))
            else:
                print('%s Error: %.3g with std of %.3g' % (cvMetric, cvMeanScores[cvIdx], cvStdScores[cvIdx]))
        print('\n')
      
    
    return cvMeanScores



#%% Defining colormap and normalization variable for plotting purposes
def mapColors(cMap_all, encodedLabels):
    
    '''
    A function that maps encoded labels to a colormap, and returns the colormap normalization variable along with
    the corresponding mapped indices.
    
    Inputs:
        cMap_all -> A list of all colors
        encodedLabels -> A NumPy array of indices, whose value points to the index of the color in cMap_all.
        
    Outputs:
        cMap_final -> Indexed colormap
        cNorm -> Normalization variable 
    '''
    
    from matplotlib import colors
    
    # Intermediary variable holding all the mapped colors based on encodedLabels
    cMap_temp = [cMap_all[j] for j in np.unique(encodedLabels)]
    
    # Creating boundaries (length n+1) for the discretized colors (length n)
    boundsFull = np.arange(-0.5,max(np.unique(encodedLabels))+1,1).tolist()
    
    # List of unique encoded labels to cycle through to identify which bounds should be stored in bounds from boundsFull
    bounds = []
    encodedLabelList = np.unique(encodedLabels).tolist()
    
    # For identification of bounds
    lowerBoundSkip = True
    
    
    # Looping through each bound in boundsFull, and identifying which demarcates the unique encoded labels in encodedLabelList
    for boundIdx, lowerBound in enumerate(boundsFull):

        # End the loop if either encodedLabelList is empty (all bounds have been found), or when the lower bound reaches
        # the index limit.
        # For the latter case, the lower bound will be appended into bounds before exiting the loop.
        if len(encodedLabelList) == 0:
            break
        if lowerBound == np.max(boundsFull):
            bounds.append(lowerBound)
            break

        # Defining upper bound
        upperBound = boundsFull[boundIdx + 1]

        # If the unique label falls within this interval, both bounds are stored in bounds (provided lowerBoundSkip is True).
        # The unique label that is used to compare is also removed.
        if lowerBound < encodedLabelList[0] < upperBound:
            if lowerBoundSkip:
                bounds.append(lowerBound)
            bounds.append(upperBound)
            encodedLabelList.pop(0)
            lowerBoundSkip = True
        # If it falls outside the interval, and nothing has been appended to bounds yet (len 0), lowerBoundSkip remains True.
        else:
            if len(bounds) > 0:
                lowerBoundSkip = False
                
    # Sorting and removing any duplicate bounds
    bounds = sorted(list(set(bounds)))
    
    # Defining the color map and associated normalization variable
    cMap_final = colors.ListedColormap(cMap_temp)
    cNorm = colors.BoundaryNorm(bounds, cMap_final.N)
    
    return cMap_final, cNorm



#%% Plotting individual hotspot scores, confusion matrix and scores for all other classes.
def plotFig1(scores, classes, hotspotClasses, cm, plotShow = True, saveFig = None):
    
    
    '''Function to plot various results concerning the performance of the model on the test set. This includes the exact
    test scores for each hotspot class, confusion matrix across all classes, and a graph of the 3 metrics (precision, recall
    and F1-score) for all classes
    
    Inputs include:
        scores --> NumPy array of performance metrics for all classes in the format produced by sklearn's
                   `precision_recall_fscore_support`.
        classes --> A NumPy Array of all unique classes in y_train.
        hotspotClasses --> A list of all unique hotspot classes as defined in the settings.
        cm --> Pure (unedited) Confusion matrix from the test scores as computed by sklearn's confusion_matrix.
        plotShow --> Boolean to indicate if the function wants to plot the figure inline (True), or not display the output plot
                     at all. Defaults to True.
        saveFig --> Filename for the resultant plot if the user wishes to save it. Defaults to None.
        
    '''
    
    import seaborn as sns
    
    ## Defining base figure and gridspec (subplot alternative).
    fig1 = plt.figure(figsize = (15,12.5))
    gs1 = gridspec.GridSpec(4, 4)
    
    
    ## METRICS (HOTSPOT METRICS) - PLOT 1a
    precisionTxt = 'Precision:'
    recallTxt = 'Recall:'
    fScoreTxt = 'F-Score:'
    
    # Looping through each hotspot class in hotspotClasses to obtain their corresponding scores for all 3 metrics.
    for hotspotClass in hotspotClasses:
        # Getting index of classFocus
        hotspotClassIdx = classes.tolist().index(hotspotClass)
        # Printing out scores of testing dataset
        precisionTxt += '  %s - %.3g' % (hotspotClass.capitalize(), scores[0][hotspotClassIdx])
        recallTxt += '  %s - %.3g' % (hotspotClass.capitalize(), scores[1][hotspotClassIdx])
        fScoreTxt += '  %s - %.3g' % (hotspotClass.capitalize(), scores[2][hotspotClassIdx])

    # Displaying the scores
    ax_hotspotMetrics = fig1.add_subplot(gs1[1,0:4])
    hotspotMetrics_properties = {'ha':'center', 'fontsize':16, 'fontweight':'bold'}
    ax_hotspotMetrics.axis([0, 10, 0, 1])
    ax_hotspotMetrics.text(5,0.8,precisionTxt, **hotspotMetrics_properties)
    ax_hotspotMetrics.text(5,0.5,recallTxt, **hotspotMetrics_properties)
    ax_hotspotMetrics.text(5,0.2,fScoreTxt, **hotspotMetrics_properties)
    ax_hotspotMetrics.axis("off")

    
    
    
    ## METRICS (CONFUSION MATRIX) - PLOT 1b
    # First normalizing confusion matrix values to be between 0 and 1.
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    ax_CM = fig1.add_subplot(gs1[2:4,0:2]) # Gridspace for the Confusion Matrix
    sns.heatmap(cm, annot=True, yticklabels=classes, xticklabels=classes, ax=ax_CM, cmap=plt.cm.Reds)
    ax_CM.set_xlabel('Predicted')
    ax_CM.set_ylabel('Ground Truth')
    ax_CM.set_title('Confusion Matrix (Normalized)')

    
    
    
    ## METRICS (INDIVUDAL LABEL METRICS) - PLOT 1c
    metricXaxis = np.arange(len(classes))
    precision = scores[0]
    recall = scores[1]
    fScore = scores[2]
    ax_metrics = fig1.add_subplot(gs1[2:4,2:4]) # Gridspace for Individual Performance Metrics
    ax_metrics.bar(metricXaxis-0.2, precision, width=0.4, label='Precision', color='blue', edgecolor='grey')
    ax_metrics.bar(metricXaxis+0.2, recall, width=0.4, label='Recall', color='green', edgecolor='grey')
    ax_metrics.plot(metricXaxis, fScore, marker='x', markersize=10, markeredgewidth=2, linestyle='', color='red', label='F-Score')
    ax_metrics.xaxis.set_ticks(metricXaxis,classes)
    ax_metrics.set_ylim(0,1.01)
    ax_metrics.set_xlabel('Classes')
    ax_metrics.set_ylabel('Performance Metrics')
    ax_metrics.set_title('Comparison of Performance Metrics & Corresponding Harmonic Means')
    ax_metrics.legend(loc='lower center')

    
    
    
    # If saveFig is defined as a file name.
    if saveFig is not None:
        plt.savefig(saveFig, bbox_inches='tight')

    # If plotShow is set to True.
    if plotShow:
        plt.show()
    else:
        plt.cla()
        plt.close(fig1)



#%% Plotting feature importances for all bands
def plotFig2(FI, features, ylimUpper = 0.3, plotShow = True, saveFig = None):
    
    '''
    Function to plot the feature importance of each of the 10 bands/features of the model that has been trained before.
    
    Inputs include:
        FI --> Feature importance matrix of the random forest classifier. Obtained by the attribute `feature_importances_`.
        features --> NumPy array of features corresponding to the 10 columns of X.
        ylimUpper --> Upper limit of y-axis. Defaults to 0.3 (Eventually displayed as 30%).
        plotShow --> Boolean to indicate if the function wants to plot the figure inline (True), or not display the output plot
                     at all. Defaults to True.
        saveFig --> Filename for the resultant plot if the user wishes to save it. Defaults to None.
        
    '''

    
    # Obtaining number of features
    nFeatures = np.arange(1,len(features)+1)
    
    # Sorting feature importance and corresponding names by descending order
    sortedFeatureImportance = np.sort(FI)[::-1]
    sortedFeatureNames = features[np.argsort(-FI)]
    
    # Defining base figure and gridspec (subplot alternative).
    fig2, ax_FI = plt.subplots(figsize=(10,4))
    
    # Creating bar chart of feature importance with corresponding feature names
    ax_FI.bar(nFeatures, sortedFeatureImportance)
    ax_FI.set_xticks(nFeatures)
    ax_FI.set_xticklabels(sortedFeatureNames)
    
    # Fixing the upper limit of the y-axis based on ylimUpper, and converting those values into readable strings as yticklabels.
    if ylimUpper >= 0.6:
        tickInterval = 0.2
    elif ylimUpper >= 0.3:
        tickInterval = 0.1
    else:
        tickInterval = 0.05
    yTickValues = np.arange(0., ylimUpper+0.01, tickInterval)
    ax_FI.set_ylim(0, ylimUpper)
    ax_FI.set_yticks(yTickValues)
    yTickStr = np.empty(yTickValues.shape, dtype=str)
    yTickStr[:] = '%'
    ax_FI.set_yticklabels(np.char.add((yTickValues * 100).astype(int).astype(str), yTickStr)) # Convert float to int to string, and append % after
    
    # Remaining plot aesthetics
    ax_FI.set_xlabel('Features', fontsize=13)
    ax_FI.set_ylabel('Importance', fontsize=13)
    ax_FI.set_title('Feature Importance', fontsize=17)
    ax_FI.tick_params('both', labelsize=11)
    
    # If saveFig is defined as a file name.
    if saveFig is not None:
        plt.savefig(saveFig, bbox_inches='tight')

    # If plotShow is set to True.
    if plotShow:
        plt.show()
    else:
        plt.cla()
        plt.close(fig2)



#%% Plotting performance of the model on full test images
def plotFig3(model, testFilename, settingStr, classes, threshold, plotShow = True, saveFig = None):
    
    '''
    Function to plot the results of the model performing on a full test image, and the associated probabilities for
    predicted hotspot class pixels.
    
    Inputs include:
        model --> The trained model.
        testFilename --> Full file path for the raster test file (.tif). Has to follow a fixed format in order for the 
                         hardcoding to work properly (i.e. ...\Data\Lava Images\S2\`eventname`\*.tif). In particular
                         `Lava Images`, `S2` and whatever was defined as `eventname` will be extracted in a hardcoded manner.
        settingStr --> A single string representing the model's settings (combination of input images). Follows the format
                       '1-S2-lava  1-S2-lava  2-LS8-lava' as an example.
        classes --> A NumPy Array of all unique classes in y_train.
        threshold -> Threshold for which probabilities that do not exceed are set to 0, and their respective majority class
                     augmented (to something random for now). Used in setThreshold (If none, no thresholding is applied).
        plotShow --> Boolean to indicate if the function wants to plot the figure inline (True), or not display the output plot
                     at all. Defaults to True.
        saveFig --> Filename for the resultant plot if the user wishes to save it. Defaults to None.
        
    '''
    
    import rasterio,rasterio.plot
    
    ## HARDCODED PORTION
    # Specific Case Name, satellite and hotspot.
        # Special condition for combined hotspot class as it will not appear on the directory (always Fire/Lava only)
    caseName = testFilename.split('\\')[-2]
    caseSatellite = testFilename.split('\\')[-3].upper()
    if 'hotspot' in classes:
        caseHotspot = 'hotspot'
    else:
        caseHotspot = testFilename.split('\\')[-4].split(' ')[0].lower()
    
    # Colormap for all 7 possible classes
    cMap_all = ['red', 'green', 'lightblue', 'grey', 'black', 'orange', 'brown']
    
    # Simple function mapping class to integers for encoding below (in either directions)
    def str_class_to_int(class_array, direction = 'str2num'):
        if direction == 'str2num':
            class_array[class_array == 'lava'] = 0
            class_array[class_array == 'vegetation'] = 1
            class_array[class_array == 'cloud'] = 2
            class_array[class_array == 'plume'] = 3
            class_array[class_array == 'shadow'] = 4
            class_array[class_array == 'fire'] = 5
            class_array[class_array == 'hotspot'] = 6
            return(class_array.astype(int))
        elif direction == 'num2str':
            class_array = class_array.astype(int).astype(str)
            class_array[class_array == '0'] = 'lava'
            class_array[class_array == '1'] = 'vegetation'
            class_array[class_array == '2'] = 'cloud'
            class_array[class_array == '3'] = 'plume'
            class_array[class_array == '4'] = 'shadow'
            class_array[class_array == '5'] = 'fire'
            class_array[class_array == '6'] = 'hotspot'
            return(class_array)
    
    
    
    
    # Reading in the raster file from testFilename, and also extracting its bounds for plotting later.
    testRasterfile = rasterio.open(testFilename,'r')
    nBands = testRasterfile.count
    boundingBox = testRasterfile.bounds
    imgExtent = (boundingBox[0], boundingBox[2], boundingBox[1], boundingBox[3]) # minX / maxX / minY / maxY [Extent for imshow]
    img = rasterio.plot.reshape_as_image(testRasterfile.read())
    testRasterfile.close()
    
    # Setting threshold for predicted labels based on its respective hotspot class (caseHotspot) probabilities
    caseHotspotTag = np.where(classes == caseHotspot)[0][0]
    if caseHotspotTag == 0:
        raise ValueError('caseHotspotTag value is 0 - cannot do masking with 0 (As all values in class_proba will be 0)')
    predictedLabels, class_proba = setThreshold(model.predict(img.reshape(-1,nBands)), model.predict_proba(img.reshape(-1,nBands)), 
                                                caseHotspot, caseHotspotTag, threshold = threshold, replacedClass = None)

    # Encoding the predicted labels with str_class_to_int and mapping them to colors based on mapColors (defined in 
    # utility_functions.py too).
    predictedLabelsEncoded = str_class_to_int(predictedLabels.copy().reshape(img[:, :, 0].shape), direction = 'str2num')
    cmap, norm = mapColors(cMap_all, predictedLabelsEncoded)


    # Obtaining predicted probabilities for only hotspot pixels (to see how confident the model was in predicting these hotspot pixels)
    classIndexVector = np.argmax(class_proba, axis=1, keepdims=True)
    classIndices = np.array(np.column_stack((np.arange(0,class_proba.shape[0]), classIndexVector)))
    class_proba = class_proba[classIndices[:,0], classIndices[:,1]].reshape(img[:, :, 0].shape) # Indexing out the max probabilities from class_proba
    classIndexBool = classIndexVector.copy().reshape(img[:, :, 0].shape) # Obtaining only the index of lava/fire pixels (boolean)
    classIndexBool[classIndexBool!=caseHotspotTag] = 0
    class_proba[classIndexBool==0] = 0 # Finally setting all non-hotspot pixels probabilities to 0           
    
    
    
    ## PLOTTING OF FIGURE
    fig3 = plt.figure(figsize = (16,12))
    gs2 = gridspec.GridSpec(2, 4)

    # Original Plot of image - PLOT 3a
    bandsViz = [6,4,3] # False color composite - SWIR2, NIR, Red
    ax_og = fig3.add_subplot(gs2[0,1:3])
    ax_og.imshow(rasterio.plot.adjust_band(img[:,:,bandsViz]), extent=imgExtent)
    ax_og.set_title('Original (%s %s Image) | %s' % (caseSatellite, caseHotspot.capitalize(), caseName))

    # Full random forest plot - PLOT 3b
    ax_rf = fig3.add_subplot(gs2[1,0:2])
    ax_rf.imshow(predictedLabelsEncoded, interpolation='none', cmap=cmap, norm=norm, extent=imgExtent)
    ax_rf.set_title('Random Forest (%s)' % (settingStr))

    # Heatmap of lava/fire pixel probabilities - PLOT 3c
    ax_hm = fig3.add_subplot(gs2[1,2:4])
    hotMap = plt.cm.get_cmap('hot').reversed() # Using a reversed heatmap (lowest value is white)
    hm = ax_hm.imshow(class_proba, interpolation='none', cmap=hotMap, extent=imgExtent, vmin=0, vmax=1) # Mapping to color range of [0,1]
    ax_hm.set_title('Probabilities of Predicted %s Pixels' % (caseHotspot.capitalize()))
    hm_cbar = fig3.colorbar(hm, ax=ax_hm)
    hm_cbar.set_label('Probability')

    # If saveFig is defined as a file name.
    if saveFig is not None:
        plt.savefig(saveFig, bbox_inches='tight')

    # If plotShow is set to True.
    if plotShow:
        plt.show()
    else:
        plt.cla()
        plt.close(fig3)




#%% Plotting the spectra (pixel value distribution) of a given data
def plotSpectra(X, y, classSet, settingStr, features, plotShow = True, saveFig = None):

    '''
    Function to plot spectral profile summary (average & min-max range) of a set of data.
    
    Inputs:
        X --> Predictor variable matrix of size (datapoints, 10).
        y --> Target variable vector of size (datapoints,)
        classSet --> A list of tuples, where each tuple contains (i) the class to plot, and (ii) the color of that class in
                     the resultant plot.
        settingStr --> A single string representing the model's settings (combination of input images). Follows the format
                       '1-S2-lava  1-S2-lava  2-LS8-lava' as an example.
        features --> NumPy array of features corresponding to the 10 columns of X.
        plotShow --> Boolean to indicate if the function wants to plot the figure inline (True), or not display the output plot
                     at all. Defaults to True.
        saveFig --> Filename for the resultant plot if the user wishes to save it. Defaults to None.
        
    Outputs:
        df --> Dataframe containing information of the mean, min, and max of each of the 10 bands/features for each class
               defined in classSet.
        
    '''
    
    from copy import deepcopy
    
    # Modifying the setting string a little to be more presentable in the figure title later on.
    settingStrModified = settingStr.replace('  ',' | ').replace('-',' ')
    figTitle = 'Training Data Spectra [%s]' % (settingStrModified)

    # Creating an array of empty data to store the class, band, mean, min and max values in the following loop.
    dataStack = np.empty((0,5))
    for Set in classSet:
        
        if Set[0] not in np.unique(y):
            raise ValueError('Defined class `ge%s` not found in the target variable y.' % (Set[0]))
        
        # Only requiring the class type in this loop, for which it is used to index out corresponding pixelData from X.
        Class = Set[0]
        pixelData = deepcopy(X[y==Class, :])

        # Obtaining the mean, min and max values of pixelData.
        Mean = np.hstack(np.apply_along_axis(np.nanmean, 0, pixelData)).reshape(-1,1)
        Min = np.apply_along_axis(np.min, 0, pixelData).reshape(-1,1)
        Max = np.apply_along_axis(np.max, 0, pixelData).reshape(-1,1)

        # Appending these values to dataStack.
        newStack = np.hstack((np.repeat(Class,len(features)).reshape(-1,1), features.reshape(-1,1), Mean, Min, Max))
        dataStack = np.append(dataStack, newStack, axis=0)

    # Formatting dataStack as a dataframe.
    df = pd.DataFrame(data=dataStack, index=None, columns=['Class','Band','Mean','Min','Max'])


    # Plotting results
    figSpectral, axSpectral = plt.subplots(figsize=(12,8))
    for Set in classSet:
        
        # Defining class and corresponding plot color.
        Class = Set[0]
        fColor = Set[1]
        
        # Plotting the mean as a line, and the shaded regions bounded by its corresponding min and max values.
        subDf = df[df['Class'] == Class].copy(deep=True)
        subDf = subDf.iloc[:,:]
        axSpectral.plot(subDf['Band'], subDf['Mean'].astype(float), marker='.', markersize=20, alpha=0.5, linestyle='-', linewidth=2,
                color=fColor, label=pd.unique(subDf['Class'])[0].capitalize())
        axSpectral.fill_between(subDf['Band'], subDf['Min'].astype(float), subDf['Max'].astype(float), facecolor=fColor, alpha=0.2)
        
    # Additional formatting/aesthetic settings for the figure
    axSpectral.legend(fontsize=18, loc='lower center')
    axSpectral.tick_params(axis='both',labelsize=15)
    axSpectral.grid(linestyle='--',alpha=0.3)
    axSpectral.set_xlabel('Bands',fontsize=20)
    axSpectral.set_ylabel('Pixel Values',fontsize=20)
    axSpectral.set_title(figTitle,fontsize=24)
    axSpectral.set_ylim(-2,2)
    axSpectral.set_yticks([-2, -1, 0, 1, 2])
    
    # If saveFig is defined as a file name.
    if saveFig is not None:
        plt.savefig(saveFig, bbox_inches='tight')

    # If plotShow is set to True.
    if plotShow:
        plt.show()
    else:
        plt.cla()
        plt.close(figSpectral)
    
    return df



#%% Setting arbitrary thresholds for images
def setThreshold(yPred, probPred, classLabel, classIdx, threshold = None, replacedClass = None):
    
    
    '''
    Augments predicted results and associated probabilities based on a given threshold
    
    Inputs:
        yPred -> Array of predicted results (m x 1)
        probPred -> Matrix of computed probabilities for each class for each datapoint (m x c)
        classLabel -> Class of interest to augment (Should be a valid element in yPred)
        classIdx -> Index of class label with respect to probPred (cth column)
        threshold -> Threshold for which probabilities that do not exceed are set to 0, and their respective majority class
                     augmented (to something random for now) [Default None - does nothing]
        replacedClass -> Class to replace the class of interest in yPred if the probability of a datapoint in probPred does 
                         not exceed the predefined threshold. [Default None - Randomly replaces with other classes]
        
    Outputs:
        yPred -> Augmented array of predicted results (m x 1)
        probPred -> Augmented matrix of computed probabilities (m x c)
    '''
    
    # In the event if threshold is None, nothing changes and the function exits
    if threshold is None:
        return (yPred, probPred)
    
    # Finding the lava pixel in each row, and if it dosent pass a certain threshold, the majority vote is negated and the 
    # probability for that pixel's hotspot class will 0.
    majorityClassIdx = np.argmax(probPred,axis=1,keepdims=True) # Array of all chosen classes for all pixels
    hotspotClassIdx = np.where(majorityClassIdx == classIdx)[0] # Getting the indices of only hotspot classes
    hotspotClassInfo = np.column_stack((hotspotClassIdx, probPred[hotspotClassIdx,classIdx])) # Stacking hotspot class indices with their probability
    toRemoveIdx = hotspotClassInfo[np.where(hotspotClassInfo[:,1] < threshold)[0],0].astype(int) # Obtaining indices of hotspot class whose probability < threshold
    probPred[toRemoveIdx,classIdx] = 0. # Zero-ing out probabilities of hotspot classes who don't meet the probability threshold

    # In the event that replacedClass is None, we set the class meant for replacement as any random classes that != classLabel
    if replacedClass is None:
        classList = np.unique(yPred)
        classList = classList[classList != classLabel]
        replacedClass = classList[0]
        
    # Replacing the classes of interest in yPred based on toRemoveIdx
    yPred[toRemoveIdx] = replacedClass
    
    return (yPred, probPred)
