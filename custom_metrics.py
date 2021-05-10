"""
    Author: Alejandro Mardones
"""
import numpy as np
from tensorflow.keras.metrics import Metric
from tensorflow import math
from tensorflow.python.keras.utils import metrics_utils
from tensorflow.python.ops import init_ops
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.utils.generic_utils import to_list

class F1Score(Metric):
    """
        Stateful F1 Score. https://keras.io/api/metrics/#as-subclasses-of-metric

        Usage examples:
            
            # Loading models using this metric
            tf.keras.models.load_model("model.h5", custom_objects={"F1Score":F1Score()})
            
            # Compiliing models using this metric
            model.compile(...,metrics=[F1Score()])
    """
    def __init__(self, name='f1_score',thresholds=None, class_id=None, dtype=None):
        super(F1Score, self).__init__(name=name, dtype=dtype)
        self.init_thresholds = thresholds
        self.class_id = class_id

        self.thresholds = metrics_utils.parse_init_thresholds(thresholds, default_threshold=0.5)
        self.true_positives = self.add_weight(name='true_positives', shape=(len(self.thresholds),), initializer=init_ops.zeros_initializer)
        self.false_positives = self.add_weight(name='false_positives',shape=(len(self.thresholds),), initializer=init_ops.zeros_initializer)
        self.false_negatives = self.add_weight(name='false_negatives', shape=(len(self.thresholds),), initializer=init_ops.zeros_initializer)   

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Accumulates true positive, false positive and false negative statistics. 
        return metrics_utils.update_confusion_matrix_variables(
            {
                metrics_utils.ConfusionMatrix.TRUE_POSITIVES: self.true_positives,
                metrics_utils.ConfusionMatrix.FALSE_POSITIVES: self.false_positives,
                metrics_utils.ConfusionMatrix.FALSE_NEGATIVES: self.false_negatives
            },
            y_true,
            y_pred,
            thresholds=self.thresholds,
            class_id=self.class_id,
            sample_weight=sample_weight)
        
    def result(self):
        # Calculates F1 Score based on the accumulated truth values
        precision = math.divide_no_nan(self.true_positives, self.true_positives + self.false_positives)
        recall = math.divide_no_nan(self.true_positives, self.true_positives + self.false_negatives)

        result = math.divide_no_nan(math.multiply(math.multiply(precision, recall), 2.0), precision + recall)
        return result[0] if len(self.thresholds) == 1 else result

    def reset_states(self):
        num_thresholds = len(to_list(self.thresholds))
        K.batch_set_value([(v, np.zeros((num_thresholds,))) for v in self.variables])

    def get_config(self):
        config = {
            'thresholds': self.init_thresholds,
            'class_id': self.class_id
        }
        base_config = super(F1Score, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))