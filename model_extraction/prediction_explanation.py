
'''
Wrapper class for storing explanations of individual timestep predictions for the extracted model
'''


class PredictionExplanation:
    def __init__(self):
        self.state_seq = []
        self.transition_explanations = []
        self.processed_subsequences = []

    def add_state(self, state):
        self.state_seq.append(state)

    def add_transition_expl(self, expl):
        self.transition_explanations.append(expl)

    def add_subsequence(self, subsequence):
        self.processed_subsequences.append(subsequence)

    def get_explanation(self):

        self.transition_explanations.append(None)
        self.processed_subsequences.append(None)

        if not (len(self.state_seq) == len(self.transition_explanations) == len(self.processed_subsequences)):
            raise ValueError("States, transitions, and subsequences have to match one-to-one")

        explanation = [(self.state_seq[i], self.transition_explanations[i], self.processed_subsequences[i])
                       for i in range(len(self.state_seq))]

        return explanation
