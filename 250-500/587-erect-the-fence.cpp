// TLE
int cross(vector<int> p, vector<int> q, vector<int> r) {
    return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0]);
}

vector<vector<int>> outerTrees(vector<vector<int>>& trees) {
    int n = trees.size();
    if(n < 4) {
        return trees;
    }
    
    vector<vector<int>> fence;
    unordered_set<int> visited;

    int begin = 0;

    for(int i = 0; i < n; i++) {
        if (trees[i][0] < trees[begin][0]) 
            begin = i;
    }
    int p = begin;

    while (true) {
        int q = (p + 1) % n;
        for(int r = 0; r < n; ++r)
        {
            if(cross(trees[p], trees[q], trees[r]) < 0)
            {
                q = r;
            }
        }
        for(int i = 0; i < n; i++)
        {
            if (visited.count(i) != 0 || i == p || i == q) {
                continue;
            }
            if (cross(trees[p], trees[q], trees[i]) == 0)
            {
                visited.insert(i);
                fence.push_back(trees[i]);
            }
        }
        if (visited.count(q) == 0) {
            fence.push_back(trees[q]);
            visited.insert(q);
        }
        if (q == begin) break;
        p = q;
    }

    return fence;
    
}

//Copied
using tree = vector<int>;
    
vector<tree> outerTrees(vector<tree>& trees) 
{
    auto cross = [](tree& B, tree& A, tree& T) -> int
    {
        return (T[1]-B[1])*(B[0]-A[0]) - (B[1]-A[1])*(T[0]-B[0]);
    };
    
    sort(trees.begin(), trees.end());
    
    deque<tree> F;
    
    for (tree T : trees)
    {
        while (F.size() >= 2 and cross(F[F.size()-1],F[F.size()-2],T) < 0)
            F.pop_back();
        F.push_back(T);

        while (F.size() >= 2 and cross(F[0],F[1],T) > 0)
            F.pop_front();
        F.push_front(T);
    }
    
    set<tree> unique(F.begin(), F.end());
    return vector<tree>(unique.begin(), unique.end());
}