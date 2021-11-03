"""
main.py


"""
import argparse
import yaml
import os,sys
from runExpt.mcImgClassExpts import fnRunImgClassMC
from runExpt.demoImgClassExpts import fnRunImgClassDemo
# ---------------------------------------------------------------------------- #
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
                if j=='channel':
                    index, temp = selectParamConfig(j, config[i][j])
                    transDict[index] = temp
                    config[i][j] = transDict
        if i == 'SimulationMode':
            simDict = {}
            index, temp = selectParamConfig(1,config[i]) # switch between Demo and Monte Carlo (MC) mode.
            simDict[index] = temp
            config[i] = simDict
        if i == 'ErrorConcealment':
            ec_dict = {}
            index, temp = selectParamConfig(1,config[i]) # switch between error concealment methods.
            ec_dict[index] = temp
            config[i] = ec_dict

    return config

def fnProcessSimulation():
    """Process configs specified in the given YAML file.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--params", help="path to the config file containing parameters", required=True)
    args = vars(ap.parse_args())

    filename = args['params']

    with open(filename) as c:
        config = yaml.load(c,yaml.SafeLoader)

    ecDict = config['ErrorConcealment']
    paramsDict = configSettings(config)

    modelDict = paramsDict['DeepModel']
    task = paramsDict['DeepModel']['task']
    splitLayer = paramsDict['SplitLayer']['split']
    splitLayerDict = paramsDict['SplitLayer']

    if 'MonteCarlo' in paramsDict['SimulationMode']:
        sim_mode = 'MonteCarlo'
        MC_runs = paramsDict['SimulationMode']['MonteCarlo']['MC_runs']
        mc_task = paramsDict['SimulationMode']['MonteCarlo']['MC_task']
        ecDict = paramsDict['ErrorConcealment']
        ec_method = [*ecDict][0]
        if ec_method == 0:
            ecDict = 'noEC'
    else:
        sim_mode = 'Demo'

    print(f'This is an {task} experiment run in {sim_mode} mode.')
    if sim_mode == 'MonteCarlo':
        print(f'We are doing the {mc_task} step.')
        if mc_task == 'LoadLossPatterns':
            if ecDict != 'noEC':
                print(f'with error concealment done with {ec_method}')
            else:
                print('No error concealment')
        else:
            print('No error concealment.')

    # dataset    = paramsDict['TestInput']['dataset']
    batch_size = paramsDict['TestInput']['batch_size']
    testdir    = paramsDict['TestInput']['testdir']
    path_base = testdir['images']

    transDict  = paramsDict['Transmission']
    outputDir = paramsDict['OutputDir']
    rowsPerPacket = transDict['rowsperpacket']
    quantization = transDict['quantization']

    channel = transDict['channel']
    if 'GilbertChannel' in channel:
        lossProbability = channel['GilbertChannel']['lossProbability']
        burstLength = channel['GilbertChannel']['burstLength']
        print(f'Gilbert-Elliott channel model selected with loss probability {lossProbability} and burst length {burstLength}')
    elif 'RandomLossChannel' in channel:
        lossProbability = channel['RandomLossChannel']['lossProbability']
        print(f'Random loss channel model selected with loss probability {lossProbability}')
    elif 'ExternalChannel' in channel:
        print('External packet traces imported')
    else:
        print('No channel model selected.')
        ecDict = 'noEC'

    if task == 'ImgClass':
        if sim_mode == 'MonteCarlo':
            fnRunImgClassMC(modelDict,splitLayerDict,ecDict,MC_runs,mc_task,batch_size,path_base,transDict,outputDir)
        if sim_mode == 'Demo':
            print('Running demo')
            fnRunImgClassDemo(modelDict,splitLayerDict,ecDict,batch_size,path_base,transDict,outputDir)

# ---------------------------------------------------------------------------- #
if __name__ == '__main__':
    # Process input yaml file and run experiment.
    fnProcessSimulation()
