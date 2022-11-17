vector<vector<int>> reconstructQueue(vector<vector<int>>& people) {
    sort(people.begin(), people.end(), 
        [](const vector<int> &p1, const vector<int> &p2)->bool
        {
            if (p1[0] == p2[0]) return p1[1] < p2[1];
            else return p1[0] > p2[0]; 
        });
    list<vector<int>> tmpQueue;
    int pos = 0;
    for (int i = 0; i < people.size(); ++i) {
        auto beg = tmpQueue.begin();
        pos = people[i][1];
        while (pos--)
            beg++;
        tmpQueue.insert(beg, people[i]);
    }
    
    return vector<vector<int>>(tmpQueue.begin(), tmpQueue.end());
}