vector<int> partitionLabels(string s) {
    vector<int> length;
    vector<vector<int>> positions;
    vector<int> record;
    for (char i = 'a'; i <= 'z'; ++i) {
        if (s.find(i) != -1) {
            record = {int(s.find(i)), int(s.rfind(i))};
            positions.push_back(record);
        }
    }

    sort(positions.begin(), positions.end(),
        [](const vector<int> &v1, const vector<int> &v2) {
            return v1[0] < v2[0];
        });
    

    int begin = 0, end = positions[0][1];

    for (int i = 1; i < positions.size(); ++i) {
        if (positions[i][0] < end) {
            end = max(positions[i][1], end);
        }
        else {
            length.push_back(end - begin + 1);
            begin = end + 1;
            end = positions[i][1];
        }
    }
    length.push_back(end - begin + 1);

    return length;
}