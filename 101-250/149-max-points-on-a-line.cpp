int maxPoints(vector<vector<int>>& points) {
    unordered_map<double, int> hash;
    int max_count = 0, same, same_x;
    for (int i = 0; i < points.size(); ++i) {
        same = same_x = 1;
        for (int j = i+1; j < points.size(); ++j) {
            if (points[i][0] == points[j][0]) {
                ++same_x;
                if (points[i][1] == points[j][1]) {
                    ++same;
                }
            }
            else {
                double dx = points[i][0] - points[j][0], dy = points[i][1] - points[j][1];
                ++hash[dy/dx];
            }
        }
        max_count = max(max_count, same_x);
        for (const auto &item : hash) {
            max_count = max(max_count, same + item.second);
        }
        hash.clear();
    }
    return max_count;
}