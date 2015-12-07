import three_moons
reload(three_moons)
import read_mnist
reload(read_mnist)
from .three_moons import three_moons
from .read_mnist import read_mnist
from .read_mnist import subsample