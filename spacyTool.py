class SpacyTool():
    def __init__(self, sentence, nlp):
        self.sentence = sentence
        # self.nlp = spacy.load('en_core_web_lg')
        self.doc = nlp(sentence)
        self.phrase = []

    def get_phrase_v1(self):
        for np in self.doc.noun_chunks:
            self.phrase.append(np.text)

        return self.phrase

    @staticmethod
    def get_phrase(nlp, sentence):
        phrase = []
        doc = nlp(sentence)
        for np in doc.noun_chunks:
            phrase.append(np.text)

        return phrase


if __name__ == '__main__':
    print(SpacyTool('room with wood floors and a stone fire place').get_phrase())
