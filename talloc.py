""" 
Task allocator for simulation.

Called by testConfig. Documentation to be sorted out later. 

"""

from modeltasks.tasks import *

def taskAllocater(task, testDir, batch_size, taskParams):
    """Chooses the task based on the user's options

    # Arguments
        task: integer value denoting the task
        testDir: directory location of the images
        batch_size: number of images to be forwarded through the model at once
        reshapeDims: list containing the dimensions of the reshaped image
        normalize: bool value indicating whether images must be normalized
    """
    if task==0:
        return CFTask(testDir['images'], taskParams['reshapeDims'], batch_size, taskParams['normalize'])
    if task==1:
        return ODTask(testDir, taskParams['reshapeDims'], batch_size, taskParams['classes'], taskParams['num_classes'])
