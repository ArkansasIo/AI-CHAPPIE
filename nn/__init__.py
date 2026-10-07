"""
nn -- a from-scratch neural network, logic, quantum and language library.

Modules
-------
Core machine learning
    linalg       : linear algebra primitives + calculus (numeric gradients)
    activations  : activation functions with analytic derivatives
    losses       : loss functions with analytic gradients
    layers       : parameterized layers (Dense, Dropout, BatchNorm) fwd/bwd
    optimizers   : SGD, Momentum, Newton, RMSProp, Adam, AdamW
    model        : Sequential container + train/predict/evaluate/save/load
    utils        : one-hot, metrics, batching, initialization, grad checking
    autograd     : reverse-mode automatic differentiation engine (Tensor)

Symbolic / discrete systems
    gates        : Boolean algebra, truth tables, smooth (differentiable) logic
    qubits       : quantum logic -- states, gates, entanglement, algorithms
    coordinates  : 3-D signed-coordinate geometry and value fields

Natural language
    grammar      : English morphology, tagging, parsing, agreement
    conjugation  : full verb conjugation paradigm (tense/aspect/mood/voice)

Algorithms
    algorithms   : every algorithm in the project in one place, with complexity
                   metadata, registry, and cross-system pipelines

The library is intentionally dependency-light (NumPy only) and readable so the
mathematics in docs/ is mirrored one-to-one by the code.
"""

from . import linalg
from . import activations
from . import losses
from . import layers
from . import optimizers
from . import model
from . import utils
from . import autograd
from . import coordinates
from . import gates
from . import qubits
from . import grammar
from . import conjugation

from .activations import ACTIVATIONS, get_activation, get_activation_grad
from .losses import LOSSES, get_loss
from .layers import Dense, Dropout, BatchNorm, Activation
from .model import Sequential, MLP
from .optimizers import OPTIMIZERS, get_optimizer
from .autograd import Tensor
from .coordinates import (
    Axis, Vec3, Values3D,
    X_AXIS, Y_AXIS, Z_AXIS, AXES, OCTANTS,
    octant_of, octant_name, all_octants,
    right_handed_frame, left_handed_frame, frame_of, handedness,
)
from .conjugation import (
    ConjugatedVerb, ConjugationTable, Subject,
    conjugate as conjugate_verb, conjugate_all,
    Person, Number, Tense, Aspect, Mood, Voice,
)

__all__ = [
    "linalg",
    "activations",
    "losses",
    "layers",
    "optimizers",
    "model",
    "utils",
    "autograd",
    "coordinates",
    "gates",
    "qubits",
    "grammar",
    "conjugation",
    "ACTIVATIONS",
    "get_activation",
    "get_activation_grad",
    "LOSSES",
    "get_loss",
    "Dense",
    "Dropout",
    "BatchNorm",
    "Activation",
    "Sequential",
    "MLP",
    "OPTIMIZERS",
    "get_optimizer",
    "Tensor",
    "Axis",
    "Vec3",
    "Values3D",
    "X_AXIS",
    "Y_AXIS",
    "Z_AXIS",
    "AXES",
    "OCTANTS",
    "octant_of",
    "octant_name",
    "all_octants",
    "right_handed_frame",
    "left_handed_frame",
    "frame_of",
    "handedness",
    "ConjugatedVerb",
    "ConjugationTable",
    "Subject",
    "conjugate_verb",
    "conjugate_all",
    "Person",
    "Number",
    "Tense",
    "Aspect",
    "Mood",
    "Voice",
]

__version__ = "1.0.0"
