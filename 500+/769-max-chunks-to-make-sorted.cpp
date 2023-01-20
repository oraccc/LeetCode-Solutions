int maxChunksToSorted(vector<int>& arr) {
    int currMax = 0, chunks = 0;
    for (int i = 0; i < arr.size(); ++i) {
        currMax = max(currMax, arr[i]);
        if (currMax == i) {
            ++chunks;
        }
    }
    return chunks;
}