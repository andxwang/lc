class Node:
    def __init__(self, val=None):
        self.val = val
        self.children = {}  # map char: node

    def __str__(self):
        lines, *_ = self._display_aux()
        return '\n'.join(lines)

    def _display_aux(self):
        """Returns list of strings, width, height, and horizontal coordinate of the root."""
        # Filter out terminal nodes ('0') from display
        display_children = {k: v for k, v in self.children.items() if k != '0'}
        
        # Check if this is an end of word
        is_end = '0' in self.children

        # No displayable children
        if not display_children:
            line = f'[{self.val or "root"}{"*" if is_end else ""}]'
            width = len(line)
            height = 1
            middle = width // 2
            return [line], width, height, middle

        # Has children
        children = sorted(display_children.items())  # Sort children for consistent display
        child_displays = [child._display_aux() for char, child in children]
        
        # Combine all children's displays
        child_lines = []
        total_width = -1  # Account for spacing between children
        for char, (lines, w, h, m) in zip(dict(children).keys(), child_displays):
            if total_width < 0:
                total_width = 0
            spaced_lines = [' ' * total_width + line for line in lines]
            child_lines.append((spaced_lines, total_width + m, char))
            total_width += w + 2  # Add spacing between children

        # Create the value line
        root_str = f'[{self.val or "root"}{"*" if is_end else ""}]'
        root_width = len(root_str)
        root_start = (total_width - root_width) // 2
        first_line = ' ' * root_start + root_str

        # Create the connecting lines
        if child_lines:
            second_line = ' ' * root_start + '┌' + '─' * (root_width - 2) + '┐'
            connector_lines = []
            for i, (lines, x, char) in enumerate(child_lines):
                spaces = x - len(connector_lines[0]) if connector_lines else x
                connector_lines = [line + ' ' * spaces + ('│' if i < len(child_lines)-1 else ' ') for line in connector_lines]
                connector_lines.append(' ' * x + f'└─{char}')
            
            # Combine all child lines
            max_child_height = max(len(lines) for lines, _, _ in child_lines)
            final_child_lines = []
            for i in range(max_child_height):
                line = ''
                for child_line, _, _ in child_lines:
                    if i < len(child_line):
                        line += child_line[i]
                    else:
                        line += ' ' * len(child_line[0])
                final_child_lines.append(line)
            
            return [first_line, second_line] + connector_lines + final_child_lines, total_width, len(connector_lines) + len(final_child_lines) + 2, total_width // 2

class Trie:

    def __init__(self):
        self.root = Node()
        
    def __str__(self):
        return str(self.root)

    def insert(self, word: str) -> None:
        curr = self.root
        for c in word:
            if c not in curr.children: 
                curr.children[c] = Node(c)
            curr = curr.children[c]
        curr.children['0'] = 1    

    def search(self, word: str) -> bool:
        curr = self.root
        for c in word:
            if c not in curr.children:
                return False
            curr = curr.children[c]
        return '0' in curr.children

    def startsWith(self, prefix: str) -> bool:
        curr = self.root
        for c in prefix:
            if c not in curr.children:
                return False
            curr = curr.children[c]
        return True
    
# Test Case 3: Multiple overlapping words
# Words: "car", "card", "care", "cargo"

# trie = Trie()
# trie.insert("car")
# trie.insert("card")
# trie.insert("care")
# trie.insert("cargo")

# print(trie)

# print("search('car')   ->", trie.search("car"))     # Expected: True
# print("search('card')  ->", trie.search("card"))    # Expected: True
# print("search('cat')   ->", trie.search("cat"))     # Expected: False
# print("startsWith('car') ->", trie.startsWith("car")) # Expected: True
# print("startsWith('ca')  ->", trie.startsWith("ca"))  # Expected: True
# print("search('cargo') ->", trie.search("cargo"))   # Expected: True

class WordDictionary:

    def __init__(self):
        self.root = {}

    def addWord(self, word: str) -> None:
        curr = self.root
        for c in word:
            if c not in curr:
                curr[c] = {}
            curr = curr[c]
        curr['*'] = True

    def search(self, word: str) -> bool:
        return self._search(self.root, word, 0)
        
    def _search(self, curr, word: str, idx: int) -> bool:
        for i in range(idx, len(word)):
            c = word[i]
            if c == '.':
                return any(self._search(curr[k], word, i + 1) for k in curr if k != '*')
            else:
                if c not in curr:
                    return False
                curr = curr[c]
        return '*' in curr
    
# Test cases for WordDictionary
word_dict = WordDictionary()

# Add words of different lengths
word_dict.addWord("cat")
word_dict.addWord("cats")
word_dict.addWord("catch")
word_dict.addWord("dog")
word_dict.addWord("dogs")

# Test 1: Exact matches
print("=== Exact Match Tests ===")
print("search('cat')  ->", word_dict.search("cat"))    # True
print("search('cats') ->", word_dict.search("cats"))   # True
print("search('cap')  ->", word_dict.search("cap"))    # False
print("search('ca')   ->", word_dict.search("ca"))     # False

# Test 2: Single wildcards at different positions
print("\n=== Single Wildcard Tests ===")
print("search('.at')  ->", word_dict.search(".at"))    # True (cat)
print("search('c.t')  ->", word_dict.search("c.t"))    # True (cat)
print("search('ca.')  ->", word_dict.search("ca."))    # True (cat)
print("search('.og')  ->", word_dict.search(".og"))    # True (dog)
print("search('d.g')  ->", word_dict.search("d.g"))    # True (dog)
print("search('.a.')  ->", word_dict.search(".a."))    # True (cat, dog)
print("search('catch.') ->", word_dict.search("catch.")) # False (too long)

# Test 3: Multiple wildcards
print("\n=== Multiple Wildcard Tests ===")
print("search('.at.') ->", word_dict.search(".at."))   # True (cats)
print("search('c...')  ->", word_dict.search("c..."))   # True (cats)
print("search('c...h') ->", word_dict.search("c...h"))  # True (catch)
print("search('..t')   ->", word_dict.search("..t"))    # True (cat)
print("search('..t..') ->", word_dict.search("..t.."))  # True (catch)
print("search('....h') ->", word_dict.search("....h"))  # True (catch)

# Test 4: Edge cases
print("\n=== Edge Cases ===")
print("search('')      ->", word_dict.search(""))       # False (empty string)
print("search('.')     ->", word_dict.search("."))      # False (no single-letter words)
print("search('......')->", word_dict.search("......")), # False (too long)
print("search('c.')    ->", word_dict.search("c."))     # False (no two-letter words)
print("search('.a..h') ->", word_dict.search(".a..h"))  # True (catch)


