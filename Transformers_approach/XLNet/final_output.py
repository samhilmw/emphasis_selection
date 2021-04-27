def final_output(text, binary=True, thres=0.5):
    """
    It converts text from the model to 
    list of words and list of probabilities 
    
    Parameters: 
    -----------
    text: str
        A string produced from the model
    
    Returns:
    --------
    word_list: list
        list of words
    prob_list: list
        list of probabilities
    """
    text = text.strip('\n')
    sent_list = text.split('\n\n')
    line_list = []
    word_list = []
    prob_list = []

    for sent in sent_list:
        line_list.append(sent.split('\n'))
    for sent in line_list:
        word = []
        prob = []
        for line in sent:
            temp = line.split('\t')
            word.append(temp[1])
            prob.append(float(temp[2]))
        word_list.append(word)
        prob_list.append(prob)

    if binary:
        lengths = [len(sent) for sent in prob_list]
        for e, length in enumerate(range(len(lengths))):
            for j in range(lengths[e]):
                if prob_list[e][j] > thres:
                    prob_list[e][j] = 1
                else:
                    prob_list[e][j] = 0


    return word_list, prob_list

if __name__ == '__main__':
    
    text = """S_0_0\tHello\t0.376
                S_0_1\tWorld\t0.854

                S_1_0\tWorld\t0.376
                S_1_1\tHello\t0.854"""
    
    words,probs =  final_output(text)
    
    print(words)
    print(probs)
