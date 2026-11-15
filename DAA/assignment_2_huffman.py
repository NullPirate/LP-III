class Node:
    def __init__(self, prob, symbol, left=None, right=None):
        self.prob = prob

        self.symbol = symbol

        self.left = left

        self.right = right

        self.code = ''

def Calculate_Probability(data):
    symbols = dict()
    for element in data:
        if symbols.get(element) == None:
            symbols[element] = 1
        else:
            symbols[element] += 1
    return symbols

def Output_Encoded(data, coding):
    encoding_output = []
    for c in data:
        print(coding[c], end = '')
        encoding_output.append(coding[c])

    string = ''.join([str(item) for item in encoding_output])
    return string

def Total_Gain(data, coding):
    before_compression = len(data) * 8
    after_compression = 0
    symbols = coding.keys()
    for symbol in symbols:
        count = data.count(symbol)
        after_compression += count * len(coding[symbol])
    print("Space usage before compression (in bits):", before_compression)
    print("Space usage after compression (in bits):", after_compression)

coding = {}

def Calculate_Codes(node, val=''):
    new_val = val + str(node.code)
    left = node.left
    right = node.right
    if left:
        Calculate_Codes(left, new_val)
    if right:
        Calculate_Codes(right, new_val)
    if not left and not right:
        coding[node.symbol] = new_val
    return coding

def Huffman_Encoding(data):
    symbol_with_probs = Calculate_Probability(data)
    symbols = symbol_with_probs.keys()
    probabilities = symbol_with_probs.values()
    print("symbols: ", symbols)
    print("probabilities: ", probabilities)

    nodes = []

    for symbol in symbols:
        nodes.append(Node(symbol_with_probs.get(symbol), symbol))

    while len(nodes) > 1:
        nodes = sorted(nodes, key=lambda x: x.prob)

        right = nodes[0]
        left = nodes[1]

        left.code = 0
        right.code = 1

        newNode = Node(left.prob+right.prob, left.symbol+right.symbol, left, right)

        nodes.remove(left)
        nodes.remove(right)
        nodes.append(newNode)

    huffman_encoding = Calculate_Codes(nodes[0])
    print(huffman_encoding)
    Total_Gain(data, huffman_encoding)
    encoded_output = Output_Encoded(data,huffman_encoding)
    print("Encoded output:", encoded_output)
    return encoded_output, nodes[0]

data = 'AAAAAAABCCCCCCDDEEEEE'
print(data)
encoded_output, huffman_tree = Huffman_Encoding(data)
