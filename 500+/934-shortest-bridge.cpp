vector<int> directions{-1, 0, 1, 0, -1};

int shortestBridge(vector<vector<int>>& grid) {
    int m = grid.size(), n = grid[0].size();
    queue<pair<int, int>> surround;
    bool flipped = false;

    for (int i = 0; i < m; ++i) {
        if (flipped) break;
        for (int j = 0; j < n; ++j) {
            if (grid[i][j] == 1) {
                dfs(surround, grid, m, n, i, j);
                flipped = true;
                break;
            }
        }
    }

    int distance = 0;
    while (!surround.empty()) {
        ++distance;
        int queue_size = surround.size();
        while (queue_size > 0) {
            auto point = surround.front();
            int r = point.first, c = point.second;
            surround.pop();
            for (int k = 0; k < 4; ++k) {
                int x = r + directions[k], y = c + directions[k+1];
                if (x >= 0 && x < m && y >= 0 && y < n) {
                    if (grid[x][y] == 2)
                        continue;
                    if (grid[x][y] == 1)
                        return distance;
                    surround.push(make_pair(x, y));
                    grid[x][y] = 2;
                }
            }
            --queue_size;
        }
    }
    return 0;
}

void dfs(queue<pair<int,int>> &surround, vector<vector<int>> &grid, int m, int n, int i, int j) {
    if (grid[i][j] == 2) return;
    if (grid[i][j] == 0) {
        surround.push(make_pair(i, j));
        grid[i][j] == 2;
        return;
    }
    grid[i][j] = 2;

    for (int k = 0; k < 4; ++k) {
        int x = i + directions[k], y = j + directions[k+1];
        if (x >= 0 && x < m && y >= 0 && y < n) {
            dfs(surround, grid, m, n, x, y);
        }
    }
}