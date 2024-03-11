def heapify(arr, n, i):
    largest = i  # 将当前节点设为最大值
    left = 2 * i + 1  # 左子节点的索引
    right = 2 * i + 2  # 右子节点的索引

    # 如果左子节点存在并且大于根节点，则更新最大值索引
    if left < n and arr[left] > arr[largest]:
        largest = left

    # 如果右子节点存在并且大于根节点，则更新最大值索引
    if right < n and arr[right] > arr[largest]:
        largest = right

    # 如果根节点不是最大值，则交换根节点和最大值节点
    if largest != i:
        arr[i], arr[largest] = arr[largest], arr[i]
        # 递归调整子树
        heapify(arr, n, largest)

def heap_sort(arr):
    n = len(arr)

    # 构建大顶堆
    for i in range(n // 2 - 1, -1, -1):
        heapify(arr, n, i)

    # 从大顶堆中逐个取出最大值并调整堆
    for i in range(n - 1, 0, -1):
        arr[0], arr[i] = arr[i], arr[0]  # 交换根节点和最后一个节点
        heapify(arr, i, 0)  # 调整堆，注意只调整前i个元素，保持后面的元素不变

if __name__ == "__main__":
    arr = [4, 8, 9, 3, 5, 0, 0, 1]
    heap_sort(arr)
    print(arr)