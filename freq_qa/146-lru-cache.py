class LinkNode:
    def __init__(self, key=None, value=None):
        self.key = key
        self.value = value
        self.prev = None
        self.next = None

class LRUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.hash_map = {}
        self.head = LinkNode()
        self.tail = LinkNode()
        self.head.next = self.tail
        self.tail.prev = self.head

    def move_node_to_tail(self, key):
        node = self.hash_map[key]
        node.prev.next = node.next
        node.next.prev = node.prev
        node.next = self.tail
        node.prev = self.tail.prev
        self.tail.prev.next = node
        self.tail.prev = node

    def get(self, key: int) -> int:
        if key not in self.hash_map: return -1
        else:
            self.move_node_to_tail(key)
            return self.hash_map[key].value


    def put(self, key: int, value: int) -> None:
        if key in self.hash_map:
            self.move_node_to_tail(key)
            self.hash_map[key].value = value
        else:
            if len(self.hash_map) == self.capacity:
                self.hash_map.pop(self.head.next.key)
                self.head.next = self.head.next.next
                self.head.next.prev = self.head
            new_node = LinkNode(key, value)
            self.hash_map[key] = new_node
            new_node.prev = self.tail.prev
            new_node.next = self.tail
            self.tail.prev.next = new_node
            self.tail.prev = new_node
            


# Your LRUCache object will be instantiated and called as such:
# obj = LRUCache(capacity)
# param_1 = obj.get(key)
# obj.put(key,value)