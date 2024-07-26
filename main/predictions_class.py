from collections import defaultdict
import heapq
import numpy as np

class Predictions:
    def __init__(self, model, q_range, unique_characters):
        """
        Initialize the Predictions class with language models, a range of n-gram sizes, and unique characters from the corpus.
        
        Parameters:
        - model (dict): Dictionary of n-gram models keyed by n-gram size.
        - q_range (list): List of n-gram sizes to use.
        - unique_characters (list): List of unique characters in the corpus.
        """
        self.model = model
        self.q_range = q_range
        self.unique_characters = unique_characters

    def _extract_contexts(self, test_word, q, missing_letter_index, with_boundaries=True):
        """
        Extract left and right contexts around the missing letter.

        Parameters:
        - test_word (str): The word with the missing letter.
        - q (int): The n-gram size.
        - missing_letter_index (int): Index of the missing letter in the word.
        - with_boundaries (bool): Whether to include boundary markers.

        Returns:
        - tuple: Left and right contexts as strings.
        """
        # Determine the maximum possible context size based on the q-gram model
        max_context_size = q - 1

        # Dynamically calculate the size of the left context based on the position of the missing letter
        left_context_size = min(missing_letter_index, max_context_size)

        # Similarly, calculate the size of the right context, taking into account the word length and position of the missing letter
        right_context_size = min(len(test_word) - missing_letter_index - 1, max_context_size)

        if with_boundaries:
            # If with_boundaries is True, include boundary markers at the start and end of the word
            test_word_with_boundaries = f"<s> {test_word} </s>"
            # Extract the left context, considering the added boundary markers and adjusted context size
            left_context = test_word_with_boundaries[max(4, missing_letter_index - left_context_size + 4):missing_letter_index + 4]
            # Extract the right context similarly, adjusting for the added boundary markers
            right_context = test_word_with_boundaries[missing_letter_index + 5:missing_letter_index + 5 + right_context_size]
        else:
            # Extract contexts without boundary markers, using the dynamically calculated context sizes
            left_context = test_word[:missing_letter_index][-left_context_size:]
            right_context = test_word[missing_letter_index + 1:][:right_context_size]

        # Return the extracted contexts, ensuring any leading or trailing whitespace is removed
        return ' '.join(left_context.strip()), ' '.join(right_context.strip())

    def _format_sequence(self, left_context, letter, right_context):
        """
        Format a sequence by combining contexts with a candidate letter.

        Parameters:
        - left_context (str): Left context around the missing letter.
        - letter (str): Candidate letter to fill the missing position.
        - right_context (str): Right context around the missing letter.

        Returns:
        - str: The formatted sequence.
        """
        # Concatenate left context, letter, and right context into a single string
        return f"{left_context} {letter} {right_context}".strip()

    def _calculate_log_probability(self, model, sequence, bos=True, eos=True):
        """
        Calculate the log probability of a sequence using a specified language model.

        Parameters:
        - model (object): The language model to use.
        - sequence (str): The sequence to evaluate.
        - bos (bool): Whether to include beginning-of-sequence marker.
        - eos (bool): Whether to include end-of-sequence marker.

        Returns:
        - float: The log probability of the sequence.
        """
        # Compute the log probability of the sequence with the given model
        return model.score(sequence, bos=bos, eos=eos)

    def _select_all_predictions(self, log_probabilities):
        """
        Select all predictions based on their log probabilities and normalize them.

        Parameters:
        - log_probabilities (dict): Dictionary of log probabilities keyed by letters.

        Returns:
        - list: List of letters sorted by their normalized probabilities.
        """
        # Convert log probabilities to actual probabilities for easier interpretation
        probabilities = {letter: np.exp(log_prob) for letter, log_prob in log_probabilities.items()}

        # Normalize the probabilities so they sum up to 1
        total_prob = sum(probabilities.values())
        normalized_probabilities = {letter: prob / total_prob for letter, prob in probabilities.items()}

        # Select all predictions sorted by normalized probability in descending order
        all_predictions = heapq.nlargest(len(normalized_probabilities), normalized_probabilities.items(), key=lambda item: item[1])
        return all_predictions

    def _get_log_probabilities(self, test_word, missing_letter_index, with_boundaries):
        """
        Get log probabilities for each candidate letter for a specific missing letter position.

        Parameters:
        - test_word (str): The word with the missing letter.
        - missing_letter_index (int): Index of the missing letter in the word.
        - with_boundaries (bool): Whether to include boundary markers.

        Returns:
        - dict: Dictionary of log probabilities keyed by letters.
        """
        # Dictionary to store log probabilities of each letter
        log_probabilities = defaultdict(list)

        for q in self.q_range:
            model = self.model.get(q)
            if model is None:
                # Skip if the model for the current n-gram size is not available
                continue

            # Extract the left and right contexts around the missing letter
            left_context, right_context = self._extract_contexts(test_word, q, missing_letter_index, with_boundaries)

            for letter in self.unique_characters:
                # Create a sequence by inserting the candidate letter between contexts
                sequence = self._format_sequence(left_context, letter, right_context)
                # Calculate the log probability of this sequence
                log_prob = self._calculate_log_probability(model, sequence, bos=with_boundaries, eos=with_boundaries)
                # Append the log probability to the corresponding letter's list
                log_probabilities[letter].append(log_prob)

        # Sum the log probabilities for each letter using logaddexp for numerical stability
        return {letter: np.logaddexp.reduce(log_probs) for letter, log_probs in log_probabilities.items()}

    def _predict_missing_letter(self, test_word, missing_letter_index, with_boundaries=True):
        """
        Predict the missing letter at a specific position using context-sensitive approach.

        Parameters:
        - test_word (str): The word with the missing letter.
        - missing_letter_index (int): Index of the missing letter in the word.
        - with_boundaries (bool): Whether to include boundary markers.

        Returns:
        - str: The most probable missing letter.
        """
        # Get log probabilities for each candidate letter for the specific missing letter position
        log_probabilities = self._get_log_probabilities(test_word, missing_letter_index, with_boundaries)
        # Select all predictions and return the most probable letter
        predictions = self._select_all_predictions(log_probabilities)
        return predictions[0][0]

    def context_sensitive(self, test_word):
        """
        Context-sensitive prediction using boundary markers for a single missing letter.

        Parameters:
        - test_word (str): The word with a single missing letter.

        Returns:
        - list: Sorted list of predicted letters and their probabilities.
        """
        # Find the index of the missing letter in the word
        missing_letter_index = test_word.index('_')
        # Get log probabilities for each candidate letter for the specific missing letter position
        log_probabilities = self._get_log_probabilities(test_word, missing_letter_index, with_boundaries=True)
        # Select all predictions and return them sorted by probability
        return self._select_all_predictions(log_probabilities)

    def context_no_boundary(self, test_word):
        """
        Context prediction without boundary markers.

        Parameters:
        - test_word (str): The word with a single missing letter.

        Returns:
        - list: Sorted list of predicted letters and their probabilities.
        """
        # Find the index of the missing letter in the word
        missing_letter_index = test_word.index('_')
        # Get log probabilities for each candidate letter for the specific missing letter position
        log_probabilities = self._get_log_probabilities(test_word, missing_letter_index, with_boundaries=False)
        # Select all predictions and return them sorted by probability
        return self._select_all_predictions(log_probabilities)

    def base_prediction(self, test_word):
        """
        Base prediction method without any context.

        Parameters:
        - test_word (str): The word with a single missing letter.

        Returns:
        - list: Sorted list of predicted letters and their probabilities.
        """
        # Find the index of the missing letter in the word
        missing_letter_index = test_word.index('_')
        # Dictionary to store log probabilities of each letter
        log_probabilities = {}

        # Format the test word to match the training format by inserting spaces between characters
        formatted_test_word = " ".join(test_word)

        # Choose the model with the largest n-gram size available
        max_q = max(self.q_range)
        model = self.model.get(max_q)
        if not model:
            return []

        for letter in self.unique_characters:
            # Create a candidate word by replacing the missing letter with the current candidate letter
            candidate_word = formatted_test_word[:missing_letter_index * 2] + letter + formatted_test_word[missing_letter_index * 2 + 1:]
            # Calculate the log probability of this candidate word
            log_probability = self._calculate_log_probability(model, candidate_word, bos=False, eos=False)
            # Store the log probability for the current letter
            log_probabilities[letter] = log_probability

        # Select all predictions and return them sorted by probability
        return self._select_all_predictions(log_probabilities)
    
    def predict_multiple_missing_letters(self, test_word, with_boundaries=True):
        """
        Predict multiple missing letters in a word using optimized beam search and log probabilities.

        Parameters:
        - test_word (str): The word with missing letters.
        - with_boundaries (bool): Whether to include boundary markers.

        Returns:
        - str: The word with predicted missing letters filled in.
        """
        # Find the indices of all missing letters in the word
        missing_letter_indices = [i for i, char in enumerate(test_word) if char == '_']

        # Initialize the beam with the original test word and an initial score of 0
        beam = [(0, test_word, missing_letter_indices)]

        # Set the initial beam width and maximum beam width
        beam_width = 6
        max_beam_width = 12

        # Set the confidence threshold for early stopping
        confidence_threshold = 0.9

        # Set the maximum number of iterations
        max_iterations = 100

        # Initialize the cache for storing log probabilities
        log_prob_cache = {}

        # Iterate until all missing letters are predicted or the maximum number of iterations is reached
        iteration = 0
        while beam and iteration < max_iterations:
            new_beam = []

            for score, word, indices in beam:
                if not indices:
                    # If no more missing letters, add the word to the new beam
                    heapq.heappush(new_beam, (score, word, indices))
                else:
                    # Select the next missing letter index to predict
                    missing_letter_index = indices[0]

                    # Check if the log probabilities are already cached
                    if (word, missing_letter_index) in log_prob_cache:
                        log_probabilities = log_prob_cache[(word, missing_letter_index)]
                    else:
                        # Get log probabilities for each candidate letter at the current missing index
                        log_probabilities = self._get_log_probabilities(word, missing_letter_index, with_boundaries)
                        log_prob_cache[(word, missing_letter_index)] = log_probabilities

                    # Sort the candidate letters by their log probabilities in descending order
                    sorted_candidates = sorted(log_probabilities.items(), key=lambda x: x[1], reverse=True)

                    # Dynamically expand the beam based on the quality of the candidates
                    expanded_beam_width = min(len(sorted_candidates), max_beam_width)

                    for letter, log_prob in sorted_candidates[:expanded_beam_width]:
                        # Create a new word by replacing the missing letter with the candidate letter
                        new_word = word[:missing_letter_index] + letter + word[missing_letter_index+1:]
                        # Calculate the new score by adding the log probability
                        new_score = score + log_prob
                        # Update the indices to predict the next missing letter
                        new_indices = indices[1:]

                        # Apply early stopping if the confidence score exceeds the threshold
                        if np.exp(log_prob) >= confidence_threshold:
                            return new_word

                        # Add the new word with its score and remaining indices to the new beam
                        heapq.heappush(new_beam, (new_score, new_word, new_indices))

            # Update the beam with the top candidates for the next iteration
            beam = heapq.nlargest(beam_width, new_beam)
            iteration += 1

        # Return the word with the highest score
        return beam[0][1]