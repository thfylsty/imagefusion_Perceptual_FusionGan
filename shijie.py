import tensorflow as tf

def _to_tensor(x, dtype):
    """Convert the input `x` to a tensor of type `dtype`.
    # Arguments
        x: An object to be converted (numpy array, list, tensors).
        dtype: The destination type.
    # Returns
        A tensor.
    """
    return tf.convert_to_tensor(x, dtype=dtype)
_EPSILON = 1e-7
def epsilon():
    """Returns the value of the fuzz factor used in numeric expressions.
    # Returns
        A float.
    # Example
    ```python
        >>> keras.backend.epsilon()
        1e-07
    ```
    """
    return _EPSILON


def set_epsilon(e):
    """Sets the value of the fuzz factor used in numeric expressions.
    # Arguments
        e: float. New value of epsilon.
    # Example
    ```python
        >>> from keras import backend as K
        >>> K.epsilon()
        1e-07
        >>> K.set_epsilon(1e-05)
        >>> K.epsilon()
        1e-05
    ```
    """
    global _EPSILON
    _EPSILON = e

def cal_entropy(target):
    #logp = K.log(p)
    #en=-tf.reduce_sum(p*logp)
    _epsilon = _to_tensor(epsilon(), target.dtype.base_dtype)
    target = tf.clip_by_value(target, _epsilon, 1. - _epsilon)
    tglogtarget=tf.log(target)/tf.log(2.)
    return - tf.reduce_sum(target * tglogtarget, -1)

def Entropy_loss(y_true,y_pred):
    en_true=cal_entropy(y_true)
    en_pred=cal_entropy(y_pred)
    chai_en=tf.abs(en_pred-en_true)
    return chai_en