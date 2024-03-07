# Install dependencies
import regex as re


class TrieNode:
    """ Used in TrieTree """
    def __init__(self):
        self.children = {}
        self.token = None


class TrieTree:
    """ Use to speed up encoding function by ~100x (even more at larger scales as well) """
    def __init__(self):
        self.root = TrieNode()
    
    def add_to_trie(self, ascii_sequence, token):
        current = self.root
        for ascii_val in ascii_sequence:
            if ascii_val not in current.children:
                current.children[ascii_val] = TrieNode()
            current = current.children[ascii_val]
        current.token = token

    def find_longest_matching_token(self, ascii_sequence):
        current = self.root
        last_token = None
        for ascii_val in ascii_sequence:
            if ascii_val in current.children:
                current = current.children[ascii_val]
                if current.token:
                    last_token = current.token
            else:
                break
        return last_token


class BPETokenizer():    

    """ 
        Create a bytepair encoding tokenizer to train your own tokenizer based on specified raw text and a desired vocabulary length. 
        Note: Only works for ASCII characters. Non-ASCII characters will get a '?'
    """

    # Defining the GPT4 Regex here for use in function
    GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

    def __init__(self, regex=GPT4_SPLIT_PATTERN):
        self.encoding_map = {}
        self.decoding_map = {}
        self.regex = regex

    def regex_setter(self, re):
        """ Simple setter function for regex """
        self.regex = re

    def encmap_getter(self):
        """ Simple getter to return our encoding map """
        return self.encoding_map
        
    def decmap_getter(self):
        """ Simple getter to return our decoding map """
        return self.decoding_map

    def _encodetext(self, text):
        """
            Takes in raw text and returns 2-D python array of ASCII values (e.g., [[70, 105, 114, 115, 116], [32, 67,...]])
            Replaces non-ASCII with '?'
        """
        if self.regex == None:
            text = [text]
        else:
            text = re.findall(self.regex, text)     # Converts text to ['First', ' Citizen', ':\n', 'Before', ' we', ' proceed',...]
        return list( list(t.encode("ascii", errors="replace")) for t in text )

    def _mergepair(self, tokens, pair_tok):
        """
            Takes in 2-D python list of lists of tokens and iterates through, replacing 'tok' where 'pair' exists
        """
        pair, tok = pair_tok
        merged_tokens = []
        for block in tokens:
            merged_block = []
            skip_next_idx = False
            if (len(block) < 2):
                merged_block = block           # Simply keep our block the same if not 2+ elements
            else:
                for idx in range(len(block) - 1):
                    if skip_next_idx:
                        skip_next_idx = False
                        if idx == len(block) - 2:
                            merged_block.append(block[idx + 1])
                    elif block[idx] == pair[0] and block[idx + 1] == pair[1]:
                        merged_block.append(tok)  # Merge pair into 'tok'
                        skip_next_idx = True  # Skip the next index as it's already merged
                        if idx == len(block) - 2:  # Check if at the end after merging
                            break  # No need to append anything else, as the pair was the last two elements
                    else:
                        merged_block.append(block[idx])
                        if idx == len(block) - 2:  # Handle the last element if it's not part of a pair
                            merged_block.append(block[idx + 1])

            merged_tokens.append(merged_block)
        return merged_tokens

    def train(self, text, vocab_size):
        """
            Takes in raw text and a desired length for max vocabulary size
            Upon completion, sets encoder_map and decoder_map to the BPE mappings (forward, backward resp.) 
        """
        encoded_text = self._encodetext(text)
        encoding_map = {}                 # Create python dictionary of merges
        num_merges = vocab_size - 128     # Number of iterations of BPE
        
        for i in range(num_merges):
            bytepair_count = {}
            for block in encoded_text:
                if len(block) > 1:
                    for idx in range(len(block)-1):
                        pair = (block[idx], block[idx+1])
                        count = bytepair_count.get(pair, 0)
                        bytepair_count[pair] = count+1
            
            # Once done iterating through all the ascii values, sort and assign most freq bytepair to new token
            freq_pair = max(bytepair_count, key=bytepair_count.get)
            new_token = 128 + i
            encoding_map[freq_pair] = new_token
            encoded_text = self._mergepair(encoded_text, (freq_pair, new_token))

        self.encoding_map = encoding_map
        self.decoding_map = {value: key for key, value in encoding_map.items()}
        self._optimize_mapping()   # Run our mapping optimizer so we can more efficiently encode / decode in the future

    def _optimize_mapping(self):
        """
            Creates self.optimized_encoding_map and self.optimized_decoding_map from their primitive parent mappings.
            These optimized mappings allow us to compute these optimized mappings once and save time on calling encode / decode:
                encode(): finds longest token that matches the initial encodings and replaces; prevents iterative process
                decode(): goes directly from token to full length of chars (rather than taking iterative steps)
        """
        optimized_decoding_map = {i: [i] for i in range(128)}
        decoding_map_list = list(self.decoding_map.items())
        for tok, pair in decoding_map_list:
            expanded_pair = [pair[0], pair[1]]
            while max(expanded_pair)>127:
                for i, pair_token in enumerate(expanded_pair):
                    expanded_pair[i:i+1] = optimized_decoding_map[pair_token]
            optimized_decoding_map[tok] = expanded_pair
        
        self.optimized_decoding_map = optimized_decoding_map
        self.optimized_encoding_map = {tuple(value): key for key, value in optimized_decoding_map.items()}
        # Defining our TrieNode as well to speed up encoding
        self.trie_tree = TrieTree()
        for ascii_sequence, token in self.optimized_encoding_map.items():
            self.trie_tree.add_to_trie(ascii_sequence, token)

    def encode(self, text):
        """
            Takes in raw text and returns tokenized text (1-D array)
        """
        encoded_text = self._encodetext(text)
        tokenized_text = []
        for block in encoded_text:
            next_idx = 0
            while next_idx < len(block):
                token = self.trie_tree.find_longest_matching_token(block[next_idx:])
                next_idx = next_idx + len(self.optimized_decoding_map[token])
                tokenized_text.append(token)
        return tokenized_text

    def decode(self, tokens):
        """
            Takes in tokenized text (1-D array) and returns raw text
        """
        token_text = []
        for i in tokens:
            token_text.extend(self.optimized_decoding_map[i])
        decoded_text = ''.join(chr(value) for value in token_text)
        return decoded_text