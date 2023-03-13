import numpy as np


def getfreqs(stream):
    """
    This function Takes a stream as an input. it gets the number of occurrences of each symbol
    in that stream,  store it in a Dictionary containing every symbol in the stream .

    Arguments:
        stream {list} -- [the input stream we want to compress]
    Returns:    
        Dict {dictionary} -- [contains the number of occurrences of each symbol in the stream,referenced by the name of the symbol]
    """

    symbols = set(stream)  # get the unique elements
    L = len(stream)
    Dict = {}  # store each symbol with its corresponding probability

    # calculate the probability of each symbol in the stream
    for s in symbols:
        freq = stream.count(s)
        Dict[s] = freq
    return Dict


def Cumfreq(symbol, dictionary):
    """
    This Function Takes as  inputs a symbol and a dictionary containing
    all the symbols that exists in our stream and their frequencies
    and returns the cumulative frequency starting from the very
    beginning of the Dictionary until that Symbol.

    Arguments:
        symbol {[type]} -- [the symbol we want to get the cumulative frequency for]
        dictionary {[type]} -- [the dictionary that contains the frequency of all symbols]
    Returns:
        p int -- [the upper bound ,which is the sum of all frequencies from the beginning of the dictionary till the specified symbol]
    """

    P = 0
    for sym in dictionary:
        P += dictionary[sym]
        if sym == symbol:
            break
    return P


def Arithmetic_encode(stream, precision=32):
    """
    The encoder function for the arithmetic code,all of its operations are supposed to be on integers

    Arguments:
        stream {list} -- [the input stream we want to encode]

    Keyword Arguments:
        precision {int} -- [the number of precision bits used in the encoder] (default: {32})

    Returns:
        code {list} -- [the binary compressed code of the input stream]
        dic  {dictionary}  -- [it contains the dictionary used to encode the stream.It includes the number of occurrences of all symbols in the stream] 
    """
    stream.append('!')  # add the end of file symbol so the that the decoder knows when to stop

    StreamSize = len(stream)

    dic = getfreqs(stream)  # output the the dictionary that contains the probabilities of  all symbols

    index = 0

    # those numbers will be used later in the scaling and emitting step of the binary bits
    full = 2 ** precision
    half = full // 2
    quarter = half // 2

    L = 0  # the lower limit of the range
    H = full  # the upper limit of the range
    trails = 0  # a variable to trace how deep i entered at the corner case of which the low and high limits are lying in the middle of the range

    code = []  # the list that will include the compressed code

    for symbol in stream:

        freqSym = dic[symbol]  # get the frequency of the symbol
        S_high = Cumfreq(symbol, dic)  # get the higher limit of this symbol
        S_low = S_high - freqSym  # get the lower limit of this symbol

        Range = H - L  # get the range of the code
        # rescale the limits to the new symbol,we are using integer division to guarantee that the output is integer 
        H = L + Range * S_high // StreamSize
        L = L + Range * S_low // StreamSize

        # creating the cases for which we will emit 0 or 1 to our final code word
        while True:  # the first two easy cases , if my range fully falls inside one block
            if H < half:  # if the full range falls in the lower half
                code.extend([0])
                code.extend([1] * trails)
                trails = 0
                # scale the lower half to be the full range
                L *= 2
                H *= 2
            elif L >= half:  # if the full range falls in the upper half
                code.extend([1])
                code.extend([0] * trails)
                trails = 0
                # scale the upper half to be the full range
                L = 2 * (L - half)
                H = 2 * (H - half)
            elif L >= quarter and H < 3 * quarter:  # if the range is split between the the upper half and the lower half
                trails += 1
                L = 2 * (L - quarter)
                H = 2 * (H - quarter)
            else:
                break
                # add the last bits before you exit the function
    trails += 1
    if L <= quarter:
        code.extend([0])
        code.extend([1] * trails)
    else:
        code.extend([1])
        code.extend([0] * trails)
    return code, dic


def Arithmetic_decode(code, dic, precision=32):
    """The decoder function for the arithmetic code,all of its operations are supposed to be on integers

    Arguments:
        code {list} -- [the binary compressed code of the input stream]
        dic {dictionary} -- [it contains the dictionary used to encode the stream.It includes the number of occurrences of all symbols in the stream]

    Keyword Arguments:
        precision {int} -- [the number of precision bits used in the decoder] (default: {32})

    Returns:
       message {list} -- [the decoder output.It's supposed to be an identical copy of the input stream]
    """
    code_size = len(code)

    stream_size = sum(
        dic.values())  # get the stream size buy summing the number of occurrences of all the symbols in the stream

    # those numbers will be used later in the scaling and emitting step of the binary bits
    full = 2 ** precision
    half = full // 2
    quarter = half // 2

    L = 0  # the lower limit of the range
    H = full  # the upper limit of the range

    val = 0  # a variable that will hold the probability which we will iterate to find the symbol using it
    indx = 1  # an index to trace the place of the last bit used to find the code.This  is due to the limited precision we have,as we cannot compute the whole floating number at once

    message = []

    while indx <= precision and indx <= code_size:  # first get the exact amount of values that can be held in the precision available , the rest will be used during the code
        if code[indx - 1] == 1:
            val = val + 2 ** (precision - indx)
        indx += 1
    flag = 1

    while flag:  # the flag will be zero if we found the end of file symbol

        for symbol in dic:  # this loop tries to find the symbol that has the range specified by the variable val 

            freqSym = dic[symbol]  # get the frequency of the symbol
            S_high = Cumfreq(symbol, dic)  # get the higher limit of this symbol
            S_low = S_high - freqSym  # get the lower limit of this symbol

            Range = H - L  # get the range of the code

            H0 = L + Range * S_high // stream_size
            L0 = L + Range * S_low // stream_size

            if L0 <= val and val < H0:
                message.extend([symbol])
                L = L0
                H = H0
                if symbol == '!':  # this condition should be satisfied at the end of the stream
                    flag = 0
                break

        while True:
            if H < half:  # if the range is in the lower half
                L *= 2
                H *= 2
                val *= 2
                if indx <= code_size:
                    val += code[indx - 1]
                    indx += 1
            elif L >= half:  # if the range is in the upper half
                L = 2 * (L - half)
                H = 2 * (H - half)
                val = 2 * (val - half)
                if indx <= code_size:
                    val += code[indx - 1]
                    indx += 1
            elif L >= quarter and H < 3 * quarter:  # if the lower limit is in the lower half na d the  upper range is in the upper half,then we will need to expand the range more
                L = 2 * (L - quarter)
                H = 2 * (H - quarter)
                val = 2 * (val - quarter)

                if indx <= code_size:  # increase the precision of pur probability until we finish all the the code bits
                    val += code[indx - 1]
                    indx += 1
            else:
                break

    message.pop()  # remove the end of stream symbol to get the original stream
    return message