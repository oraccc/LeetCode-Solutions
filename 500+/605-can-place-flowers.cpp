bool canPlaceFlowers(vector<int>& flowerbed, int n) {
    if (n == 0) return true;
    vector<int> newBed = {0, 0};
    newBed.insert(newBed.end() - 1, flowerbed.begin(), flowerbed.end());

    for (int i = 1; i < newBed.size() - 1;) {
        if (newBed[i-1]==0 && newBed[i]==0 && newBed[i+1]==0) {
            newBed[i] = 1;
            --n;
            i += 2;
        }
        else i += 1;
        if (n == 0) return true;
    }

    return false;
}