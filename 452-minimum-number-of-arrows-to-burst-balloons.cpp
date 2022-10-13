int findMinArrowShots(vector<vector<int>>& points) {
    if (points.size() == 0) return 0;
    int num = 1;
    sort(points.begin(), points.end(),
        [](const vector<int> &b1, const vector<int> &b2) {
            return b1[1] < b2[1];
        });
    vector<int> prev = points[0], curr;
    for (int i = 1; i < points.size(); ++i) {
        curr = points[i];
        if (curr[0] > prev[1]) {
            ++num;
            prev = curr;
        }
    }
    return num;
}