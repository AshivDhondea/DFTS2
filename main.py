"""
Main script for DFTS experiments.

# Changelog:
    4 June 2020: Using SafeLoader for YAML and commented out errno code.
    7 August 2020: Fixed commenting and docstrings.
"""
import argparse
import re
import yaml
import sys
import os
import importlib
from talloc import taskAllocater
from spimulation.evalloc import evalAllocater
from spimulation.testConfig import runSimulation
#from download.utils import downloadModel # Not used, so commented out.

class ParserError(Exception):
    """
    Used to throw exceptions when multiple options are selected by user.
        
    Unchanged from original DFTS.
    
    """
    
    def __init__(self, message, errors):
        """
        Error handling.

        Parameters
        ----------
        message : TYPE
            DESCRIPTION.
        errors : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        super().__init__(message)
        self.errors = errors

def isURL(s):
    """
    Check if the given string is a valid http/https url.

    Parameters
    ----------
    s: Input string, can be a url

    Returns
    -------
    bool value stating whether the given string is a url
    """
    url = re.compile("""http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\), ]|
                        (?:%[0-9a-fA-F][0-9a-fA-F]))+""")
    return bool(url.match(s))

def selectParamConfig(p, paramDict):
    """
    Throw everything except for the user selected parameter.

    Parameters
    ----------
    p: parameter for which selection is being made
    paramDict: dictionary containing user selected parameters

    Returns
    -------
    Selected parameter and the corresponding values

    Raises
    -------
    ParserError: if more than one option is selected for a parameter
    """
    sum = 0
    index = 0

    for i in paramDict:
        sum += paramDict[i]['include']
        if paramDict[i]['include']:
            index = i
    try:
        if sum>1:
            raise ParserError("Multiple configurations selected for {}".format(p), sum)
    except Exception as e:
        raise
    else:
        if sum==0:
            return (index, False)
        else:
            return (index, paramDict[index])

def configSettings(config):
    """
    Refine the parameter dictionary to only include user selected parameters.

    Parameters
    ----------
    config: dictionary read from the YAML file

    Returns
    ----------
    Dictionary containing only the options selected by the user
    """
    for i in config:
        if i=='Transmission':
            for j in config[i]:
                transDict = {}
                if j=='channel' or j=='concealment':
                    index, temp = selectParamConfig(j, config[i][j])
                    transDict[index] = temp
                    config[i][j] = transDict
    return config

def customImport(modules, classes, functions):
    """
    Import modules, classes and functions.

    Parameters
    ----------
    modules : TYPE
        DESCRIPTION.
    classes : TYPE
        DESCRIPTION.
    functions : TYPE
        DESCRIPTION.

    Returns
    -------
    custDict : TYPE
        DESCRIPTION.

    """
    custDict = {}
    for i in range(len(modules)):
        module = importlib.import_module(modules[i])
        if functions[i]:
            myClass = getattr(module, classes[i])()
            myClass = getattr(myClass, functions[i])
            custDict[functions[i]] = myClass	
            continue
		
        myClass = getattr(module, classes[i]) 
        custDict[classes[i]] = myClass
        return custDict


def userInterface():
    """
    Read in the YAML config file and call the corresponding functions.
    
    Called by main function.

    Returns
    -------
    None.

    """
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--params", help="path to the config file containing parameters", required=True)

    args = vars(ap.parse_args())

    fileName = args['params']

    with open(fileName) as c:
        config = yaml.load(c,yaml.SafeLoader) # added SafeLoader. 4 June 2020
    paramsDict = configSettings(config)

    ############################################
    #Keras model params
    ############################################
    modelPath     = paramsDict['Model']['kerasmodel']
    customObjects = paramsDict['Model']['customObjects']
    customObjects = customImport(customObjects['module'], customObjects['class'], customObjects['functions'])

    modelDict     = {'xception':'Xception', 'vgg16':'VGG16', 'VGG19':'VGG19', 'resnet50':'ResNet50',
                     'inceptionv3':'InceptionV3', 'inceptionresnetv2':'InceptionResnetV2',
                     'mobilenet':'MobileNet', 'densenet':'DenseNet','nasnet':'NASNet'}

    if os.path.isfile(modelPath):
        model = modelPath
    elif modelPath.lower() in modelDict:
        model = modelDict[modelPath.lower()]
    else:
        print('Unable to load the given model!')
        sys.exit(0)

    ############################################
    #Task selection
    ############################################
    task = paramsDict['Task']['value']
    epoch = paramsDict['Task']['epochs']

    #############################################
    # Test input paramters                      #
    #############################################
    dataset    = paramsDict['TestInput']['dataset']
    batch_size = paramsDict['TestInput']['batch_size']
    testdir    = paramsDict['TestInput']['testdir']

    #############################################
    # Split layer
    #############################################
    splitLayer = paramsDict['SplitLayer']['split']

    #############################################
    # Transmission parameters
    #############################################
    transDict  = paramsDict['Transmission']

    simDir = paramsDict['OutputDir']['simDataDir']

    #############################################
    # Test parameters such as metrics, reshape dimensions
    #############################################
    tParam = paramsDict['taskParams']
    with open(tParam) as c:
        tConfig = yaml.load(c,yaml.SafeLoader) # added SafeLoader. 4 June 2020.
    taskParams = tConfig[task]


    evaluator = evalAllocater(task, taskParams['metrics'], taskParams['reshapeDims'], taskParams['num_classes'])
    task = taskAllocater(task, paramsDict['TestInput']['testdir'], batch_size,
                        taskParams)

    # commented out on 04 June 2020. Hans
    #if not os.path.exists(simDir):
    #    try:
    #        os.makedirs(simDir)
    #    except OSError as exc:
    #        if exc.errno != errno.EXIST:
    #            raise

    runSimulation(model, epoch, splitLayer, task, modelDict, transDict, simDir, customObjects, evaluator)

if __name__ == "__main__":
    userInterface()
