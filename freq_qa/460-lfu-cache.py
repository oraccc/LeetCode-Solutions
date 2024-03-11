class LinkNode:
    def __init__(self, key=-1, value=-1):
        self.key = key
        self.value = value
        self.prev = None
        self.next = None
        self.freq = 1

class DoubleLink:
    def __init__(self):
        self.head = LinkNode()
        self.tail = LinkNode()
        self.head.next = self.tail
        self.tail.prev = self.head
    
    def add_before_tail(self, node):
        node.prev = self.tail.prev
        node.next = self.tail
        self.tail.prev.next = node
        self.tail.prev = node

    def remove_node(self, node):
        node.prev.next = node.next
        node.next.prev = node.prev
        node.next = None
        node.prev = None
        return node

    def remove_first(self):
        return self.remove_node(self.head.next)
    
    def is_empty(self):
        return self.head.next == self.tail

class LFUCache:

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.min_freq = 0
        self.freq_map = {}
        self.node_map = {}

    def increase_freq(self, node):
        freq = node.freq
        link = self.freq_map[freq]
        link.remove_node(node)
        if link.is_empty():
            self.freq_map.pop(freq)
            if freq == self.min_freq:
                self.min_freq += 1
        node.freq += 1
        self.add_node(node)

    def add_node(self, node):
        freq = node.freq
        if freq not in self.freq_map:
            self.freq_map[freq] = DoubleLink()
        link = self.freq_map[freq]
        link.add_before_tail(node)
        self.freq_map[freq] = link

    def get(self, key: int) -> int:
        if key not in self.node_map: return -1
        node = self.node_map[key]
        self.increase_freq(node)
        return node.value

    def put(self, key: int, value: int) -> None:
        if key in self.node_map:
            node = self.node_map[key]
            node.value = value
            self.increase_freq(node)
            return
        elif len(self.node_map) == self.capacity:
            link = self.freq_map[self.min_freq]
            node = link.remove_first()
            self.node_map.pop(node.key)
        node = LinkNode(key, value)
        self.add_node(node)
        self.node_map[key] = node
        self.min_freq = 1



# Your LFUCache object will be instantiated and called as such:
# obj = LFUCache(capacity)
# param_1 = obj.get(key)
# obj.put(key,value)