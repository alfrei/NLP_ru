
class SpellChecker():
    
    """
    простая проверка правописания 
    """
    
    def __init__(self, counts):
        """
        counts : словарь частотности слов (вероятности)
        """
        self.counts = counts
        self.alphabet = 'абвгдеёжзийклмнопрстуфхцчшщъыьэюя'
    
    def known(self, words):
        """
        Return the subset of words that are actually 
        in counts dictionary.
        """
        return {w for w in words if w in self.counts}
    
    def edits0(self, word): 
        return [word]
    
    def splits(self, word):
        return [(word[:i], word[i:]) 
                for i in range(len(word)+1)]
    
    def edits1(self, word):
        """
        Return all strings that are one edit away 
        from the input word.
        """
        
        pairs      = self.splits(word)
        deletes    = [a+b[1:]           for (a, b) in pairs if b]
        transposes = [a+b[1]+b[0]+b[2:] for (a, b) in pairs if len(b) > 1]
        replaces   = [a+c+b[1:]         for (a, b) in pairs for c in self.alphabet if b]
        inserts    = [a+c+b             for (a, b) in pairs for c in self.alphabet]
        print(set(deletes + transposes + replaces + inserts))
        return set(deletes + transposes + replaces + inserts)
    
    def edits2(self, word):
        """Return all strings that are two edits away 
        from the input word.
        """
        return {e2 for e1 in self.edits1(word) for e2 in self.edits1(e1)}
    
    def correct(self, word):
        """
        Get the best correct spelling for the input word
        """
        # Priority is for edit distance 0, then 1, then 2
        # else defaults to the input word itself.
        candidates = (self.known(self.edits0(word)) or 
                      self.known(self.edits1(word)) or 
                      self.known(self.edits2(word)) or 
                      [word])
        return max(candidates, key=self.counts.get)
    