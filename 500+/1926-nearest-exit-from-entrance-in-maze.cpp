int nearestExit(vector<vector<char>>& maze, vector<int>& entrance) {
    vector<int> directions{-1, 0, 1, 0, -1};
    int m = maze.size(), n = maze[0].size();
    maze[entrance[0]][entrance[1]] = '+';

    queue<vector<int>> path;
    path.push(entrance);

    int level = 0;
    while (!path.empty()) {
        ++level;
        int pathSize = path.size();
        while (pathSize--) {
            int posX = path.front()[0];
            int posY = path.front()[1];
            path.pop();
            for (int i = 0; i < 4; ++i) {
                int posR = posX + directions[i];
                int posC = posY + directions[i+1];
                if (posR >= 0 && posR < m && posC >= 0 && posC < n && maze[posR][posC] != '+') {
                    if (posR == 0 || posR == m-1 || posC == 0 || posC == n-1) {
                        return level;
                    }
                    else {
                        maze[posR][posC] = '+';
                        path.push({posR, posC});
                    }
                }
            }
        }
    }
    return -1;
}