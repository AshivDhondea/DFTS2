"""
Evaluation allocator.

Documentation to be sorted out later.

"""

from modeltasks.eval import *

def evalAllocater(task, metrics, reshapeDims, n_classes):
	"""Returns an evaluator based on the specified task.

	# Arguments
		task: integer value denoting the task
		metrics: evaluation metrics related to that task
		reshapeDims: reshape dimensions of the image
		n_classes: integer value denoting the number of classes

	# Returns
		An evaluator object.

	"""
	if task==0:
		return CFeval(metrics, reshapeDims, n_classes)
	elif task==1:
		return ODeval(metrics, reshapeDims, n_classes)