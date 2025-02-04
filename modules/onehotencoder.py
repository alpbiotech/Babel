""" 
    ------------------- Encoder for Mousify --------------------------
    Takes a sequence or an array of sequences and encodes them
    --------------------- One Hot Encoding ---------------------------
    Simple encoding that converts each amino acid into a size 20 array.
    Array will be 1 at positions that correspond to the amino acid and 
    0 at all other positions. Encoding can either be stored as a 20xN
    matrix or a 20N vector.
"""

import re
from typing import Optional

import numpy as np
import numpy.typing as npt


from abnumber import Chain
from abnumber.exceptions import ChainParseError
from functools import partial
from multiprocessing import Pool
from tqdm import tqdm

from modules.encoder import Encoder


class OneHot(Encoder):
    """
    ## One Hot Encoding
    Simple encoding that converts each amino acid into a size 20 array.
    Array will be 1 at positions that correspond to the amino acid and
    0 at all other positions. Encoding can either be stored as a 20xN
    matrix or a 20N vector.
    """

    ALL_AMINO_ACIDS = {
        "A": 0,
        "R": 1,
        "N": 2,
        "D": 3,
        "C": 4,
        "E": 5,
        "Q": 6,
        "G": 7,
        "H": 8,
        "I": 9,
        "L": 10,
        "K": 11,
        "M": 12,
        "F": 13,
        "P": 14,
        "S": 15,
        "T": 16,
        "W": 17,
        "Y": 18,
        "V": 19,
        "X": 20,
    }

    def __init__(self, sequence: Optional[npt.ArrayLike] = None):
        super().__init__()
        if sequence is not None:
            self.sequence = sequence

    def encode(
        self, pad_size: int, flatten: bool = False, normalize_encoding: bool = False
    ) -> npt.ArrayLike:
        """
        ## Encodes the sequence or array of sequences as a one-hot encoding.
        ### Args:
                \tpad_size {int} -- Lenght of sequence padding
                \tflatten {bool} -- Whether or not the output should be flattened.
        ### Returns:
                \t {npt.ArrayLike}: Array of one-hot encoded amino acid sequence
        """
        # Check if a str or list of str was passed
        self.check_sequence()
        # Initialize array
        encoded = np.zeros(
            (len(self.sequence), pad_size, len(self.ALL_AMINO_ACIDS)), dtype=np.int8
        )

        # Populate array
        for i, sequence in enumerate(self.sequence):
            try:
                sequence_numbered = Chain(sequence, "imgt")
            except ChainParseError:
                continue
            sequence = sequence_numbered.seq

            if normalize_encoding:
                encoded = self.populate_array_normalized(
                    array_index=i,
                    encoding_array=encoded,
                    chain=sequence_numbered,
                )
            else:
                encoded = self.populate_array(
                    array_index=i,
                    encoding_array=encoded,
                    sequence=sequence,
                    pad_size=pad_size,
                )

        # Remove sequences that threw an exception
        remove_index = []
        for i, sequence in enumerate(encoded):
            for position in sequence:
                if np.sum(position) == 0:
                    remove_index.append(i)
                    break

        encoded = np.delete(encoded, remove_index, axis=0)

        if flatten:
            return self.flatten(encoded)

        return encoded

    def populate_array(
        self,
        array_index: int,
        encoding_array: npt.ArrayLike,
        sequence: str,
        pad_size: int,
    ) -> npt.ArrayLike:
        """
        ## Populates the given array and sequence in the order they appear
        This is equivaltent to "regular" one-hot encoding where the padding symbol appears
        at the end of the sequence.
        ### Args:
            array_index {int} -- At what index to populate the encoding if multiple
            sequences are passsed \n
            encoding_array {array} -- Array populated with zeros to be filled in \n
            sequence {str} -- String representation of the antibody sequence \n
            pad_size {int} -- Up to what position to pad the encoding \n
        ### Returns:
            array -- One-hot encoded array with padding at the end
        """
        # Pad at the end if sequence is less than pad_size in length
        if len(sequence) < pad_size:
            difference = pad_size - len(sequence)
            sequence = sequence + "X" * difference
        # Populate array
        for j, res in enumerate(sequence):
            try:
                encoding_array[array_index][j][self.ALL_AMINO_ACIDS[res]] = 1
            # Skip the sequence if there is an IndexError
            except IndexError:
                continue
        return encoding_array

    def populate_array_normalized(
        self,
        array_index: int,
        encoding_array: npt.ArrayLike,
        chain: Chain,
    ) -> npt.ArrayLike:
        """
        ## Populates the array in a normalized fashion according to IMGT numbering
        In this encoding scheme, every framework and CDR starts at the same position in the encoding
        and is therefore considered "normalized". The padding symbol "X" is added to positions where
        no residue is found in the numbering (E.g. pos 10 in FWR1 or the central positions of CDRs).
        In this encoding the last vector never uses the padding symbol.
        ### Args:
            array_index {int} -- At what index to populate the encoding if multiple
            sequences are passsed \n
            encoding_array {array} -- Array populated with zeros to be filled in \n
            chain {Chain} -- A Chain representation of the sequence as provided by AbNumbering \n
            pad_size {int} -- Up to what position to pad the encoding \n
        ### Returns:
            array -- IMGT normalized one-hot encoded array with padding in the CDRs
        """
        # Positions can desynchronize in CDR3 when there are insertions
        additional_position = 0
        # Need to keep track of CDR3 lenght to know when FWR4 starts
        cdr3_lenght = len(chain.cdr3_seq)
        # This list keeps track of what residue is at what position
        position_tracker = []

        # Go through each imgt numbered position
        for imgt_position, residue in chain:
            try:
                # Convert the imgt position from a position object to a string
                imgt_position_string = str(imgt_position)[1:]

                # FWR4 starts at position 118 in imgt, so we want to adjust its number
                if imgt_position_string == "118" and cdr3_lenght < 15:
                    additional_position += 15 - cdr3_lenght

                # Convert imgt position to int. This fails for some CDR3 positions
                encoding_position = int(imgt_position_string) + additional_position
                position_tracker.append((encoding_position, residue))

            # If the conversion to int fails
            except ValueError as error:
                # See the string matches the imgt position numbering standard for CDR3 (e.g. 112A)
                if bool(re.match("[A-Z]", str(imgt_position)[-1])):
                    # Remove the letter at the end of the string and add 1 to the position
                    encoding_position = (
                        int(str(imgt_position)[1:-1]) + additional_position
                    )
                    additional_position += 1
                    position_tracker.append((encoding_position, residue))

                # If the ValueError was raised for a different reason than expected
                else:
                    raise error

        # Create a set of the positions in the position tracker
        position_set = set([item[0] for item in position_tracker])
        all_position_set = set(
            range(1, 131)
        )  # Reference set to figure out what is missing
        missing_positions = all_position_set.difference(
            position_set
        )  # Figure out the difference of the sets

        # Add the missing positions to the position tracker
        for position in missing_positions:
            position_tracker.append((position, "X"))

        # Populate the array
        for j, res in position_tracker:
            try:
                encoding_array[array_index][j - 1][self.ALL_AMINO_ACIDS[res]] = 1
            # Skip the sequence if there is an IndexError
            except IndexError:
                continue

        return encoding_array

    def encode_single_process(
        self,
        batch_of_sequences: npt.ArrayLike,
        pad_size: int,
        normalize_encoding: bool = False,
        return_removed_indices: bool = False,
    ):
        """
        ## Performs encoding on a single thread
        """
        # Initialize array
        encoded = np.zeros(
            (len(batch_of_sequences), pad_size, len(self.ALL_AMINO_ACIDS)),
            dtype=np.int8,
        )

        # Keep track of what to remove later if something is wrong with the input
        remove_index = []
        # Populate array
        for i, sequence in enumerate(batch_of_sequences):
            # Sometimes <NA> remains in the dataset
            if ">" in sequence or "<" in sequence:
                remove_index.append(i)
                continue
            try:
                sequence_numbered = Chain(sequence, "imgt")
            except ChainParseError:
                remove_index.append(i)
                continue
            sequence = sequence_numbered.seq

            if normalize_encoding:
                encoded = self.populate_array_normalized(
                    array_index=i,
                    encoding_array=encoded,
                    chain=sequence_numbered,
                )
            else:
                encoded = self.populate_array(
                    array_index=i,
                    encoding_array=encoded,
                    sequence=sequence,
                    pad_size=pad_size,
                )

        # Remove sequences that threw an exception
        remove_index = []
        encoded = np.delete(encoded, remove_index, axis=0)
        if return_removed_indices:
            return encoded, remove_index
        return encoded

    def encode_multiprocess(
        self,
        pad_size: int,
        ncpus: int,
        flatten: bool = False,
        normalize_encoding: bool = False,
        return_removed_indices: bool = False,
    ) -> npt.ArrayLike:
        """
        ## Performs encode with multiple processes
        ### Args:
                \tpad_size {int} -- Lenght of sequence padding
                \tncpus {int} -- Number of processes
                \tflatten {bool} -- Whether or not the output should be flattened.
        ### Returns:
                \t {npt.ArrayLike}: Array of one-hot encoded amino acid sequence
        """
        # Check if a str or list of str was passed or if there are any out-of-the-ordinary sequences
        self.check_sequence()
        # Split data
        split_sequences = np.array_split(self.sequence, ncpus * 20)
        len_split_sequences = [len(item) for item in split_sequences]

        partial_single_process_encoding = partial(
            self.encode_single_process,
            pad_size=pad_size,
            normalize_encoding=normalize_encoding,
            return_removed_indices=return_removed_indices,
        )
        all_results = []
        with Pool(processes=ncpus) as pool:
            results = list(
                tqdm(
                    pool.imap(partial_single_process_encoding, split_sequences),
                    total=len(split_sequences),
                )
            )
            for encodings in results:
                all_results.append(encodings)

        # Need to separate results first if return_removed_indices is True
        if return_removed_indices:

            # Encodings are the first element, removed indices the second element
            encoded = [element[0] for element in all_results]
            packaged_removed_indices = [element[1] for element in all_results]

            # Removed indices are still packaged by the order they were processed
            removed_indices = []
            # We kept track of how large of a chunk was passed to each worker
            for index in range(len(len_split_sequences)):
                if index > 0:
                    # When we reindex the removed indices we need to keep track on
                    # how much space did the previous worker take in the results
                    chunk_size = len_split_sequences[index - 1]
                else:
                    # We need to define chunk_size to avoid an exception at index = 0
                    chunk_size = 0
                for item in packaged_removed_indices[index]:
                    # Reindex by shifting the result by the previous chunk_size * the position
                    reindexed_remove_index = item + (index * chunk_size)
                    removed_indices.append(reindexed_remove_index)

            encoded = np.concatenate(encoded)
            if flatten:
                return self.flatten(encoded), removed_indices
            return encoded, removed_indices

        encoded = np.concatenate(all_results)

        if flatten:
            return self.flatten(encoded)

        return encoded

    def pad(self, sequences: npt.ArrayLike, length: int) -> list:  # type: ignore
        """
        ## Pads the sequences provided to the length indicated.
        ### Args:
            sequences {list} -- List of sequences
            length {int} -- Length to be padded to. Must be greater or equal to the
            longest sequence in sequences.
            alignment {str} -- Defines whether padding is started on the left or
            the right of the sequence
        ### Returns:
            {list} -- List of padded sequences
        """
        # Initiate list
        padded_sequences = []

        for sequence in sequences:  # type: ignore
            # Find by how many lines to pad by
            difference = length - len(sequence)
            # Make sure that provided padding length makes sense
            assert (
                difference >= 0
            ), f"Padding length ({length}) provided is smaller than \
            one of the sequences ({len(sequence)})!"

            if difference > 0:
                sequence = np.append(sequence, np.zeros((difference, 21)), axis=0)

            padded_sequences.append(sequence)

        return padded_sequences

    def check_sequence(
        self,
    ) -> None:
        """
        ## Checks if the sequence is a str or list of str
        Transforms str into list of string to avoid bugs.
        """
        if isinstance(self.sequence, str):
            self.sequence = [self.sequence]

    @staticmethod
    def flatten(sequences: npt.ArrayLike) -> npt.ArrayLike:
        """
        ## Flattens the provided array of sequences.
        Assumes sequences have been padded or are the same length.
        ### Args:
                    \tsequence {npt.ArrayLike} -- Array of sequences to be flattened
        ### Returns"
                    \t {npt.ArrayLike} -- Array of flattened sequences
        """
        output = []

        for sequence in sequences:  # type: ignore
            output.append(sequence.flatten())

        return np.array(output).astype(int)
