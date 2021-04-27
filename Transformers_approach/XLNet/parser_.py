from nltk import sent_tokenize, word_tokenize

def parse_text(text):
    """
    It converts short text or passage into 
    format specific to input for the model 
    
    Parameters: 
    -----------
    text: str
        A string of input text
    
    Returns:
    --------
    result: str
        Formated text
    """
    result = ''
    sent_tokens = sent_tokenize(text)
    for sent_no,sent in enumerate(sent_tokens):
        word_tokens = word_tokenize(sent)
        for word_no,token in enumerate(word_tokens):
            result += 'S_'+str(sent_no)+'_'+str(word_no)+'\t'+token+'\n'
        result += '\n'
    result = result[:-1] 
    return result

if __name__ == '__main__':
    
    text = """A security token is a peripheral device used to gain access to an electronically restricted resource.\n
    The token is used in addition to or in place of a password.\n
    It acts like an electronic key to access something."""

    print(parse_text(text))
