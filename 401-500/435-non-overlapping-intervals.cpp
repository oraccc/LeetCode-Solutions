//Solution: Greedy Algorithm

int eraseOverlapIntervals(vector<vector<int>>& intervals) {
    int num = 0;
    sort(intervals.begin(), intervals.end(), 
        [](const vector<int> &v1, const vector<int> &v2) {
            return v1[1] < v2[1];
        });
    
    vector<int> lastGap = intervals[0], currGap;
    for (int i = 1; i < intervals.size(); ++i) {
        currGap = intervals[i];
        if (lastGap[1] <= currGap[0]) {
            lastGap = currGap;
        }
        else ++num;
    }
    return num;
}