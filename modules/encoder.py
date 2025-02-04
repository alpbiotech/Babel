""" 
    ------------------- Encoder for Mousify --------------------------
    Takes a sequence or an array of sequences and encodes them
    ------------------------------------------------------------------
"""


from abc import ABC, abstractmethod
from typing import Union
import numpy.typing as npt


class Encoder(ABC):

    @abstractmethod
    def encode(self,pad_size: int, flatten: bool = False) -> npt.ArrayLike:
        """
        ## Encodes the sequence or array of sequences as a one-hot encoding.
        ### Args:
                \tpad_size {int} -- Lenght of sequence padding
                \tflatten {bool} -- Whether or not the output should be flattened.
        ### Returns:
                \t {npt.ArrayLike}: Array of one-hot encoded amino acid sequence
        """
        pass

    @abstractmethod
    def encode_multiprocess(self, pad_size: int, ncpus: int, flatten: bool = False) -> npt.ArrayLike:
        """
        ## Encodes the sequence or array of sequences as a one-hot encoding with multiprocessing.
        ### Args:
                \tpad_size {int} -- Lenght of sequence padding \n
                \tflatten {bool} -- Whether or not the output should be flattened \n
                \tncpus {int} -- Number of CPUs to use for multiprocessing
        ### Returns:
                \t {npt.ArrayLike}: Array of one-hot encoded amino acid sequence
        """
        pass

    def set_sequence(self, sequence: Union[list[str], str, npt.ArrayLike]) -> None:
        """
        ## Sets the sequence attribute.
        Sequence can be a single sequence in a list or a list/array of sequences.
        ### Args:
                \tsequence {list-like[str]} -- Amino acid sequence
        """
        self.sequence = sequence