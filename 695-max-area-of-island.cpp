// Stack DFS

int maxAreaOfIsland(vector<vector<int>>& grid) {
    vector<int> directions = {-1, 0, 1, 0, -1};
    int max_row = grid.size(), max_col = max_row ? grid[0].size() : 0;
    int local_area = 0, area = 0;
    int x, y;
    for (int i = 0; i < max_row; ++i) {
        for (int j = 0; j < max_col; ++j) {
            if (grid[i][j] == 1) {
                local_area = 1;
                grid[i][j] = 0;
                stack<pair<int, int>> island;
                island.push(make_pair(i, j));
                while (!island.empty()) {
                    auto pos = island.top();
                    island.pop();
                    for (int k = 0; k < 4; ++k) {
                        x = pos.first + directions[k];
                        y = pos.second + directions[k+1];
                        if (x >= 0 && x < max_row && y >= 0 && y < max_col && grid[x][y] == 1) {
                                grid[x][y] = 0;
                                ++local_area;
                                island.push(make_pair(x, y));
                            }
                    }
                }
                area = max(area, local_area);
            }
        }
    }
    return area;
}

