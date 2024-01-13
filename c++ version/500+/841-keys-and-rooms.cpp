bool canVisitAllRooms(vector<vector<int>>& rooms) {
    int n = rooms.size();
    vector<bool> visited(n, false);
    dfs(0, rooms, visited);
    if (count(visited.begin(), visited.end(), true) == n) {
        return true;
    }
    else return false;
}

void dfs(int i, vector<vector<int>> &rooms, vector<bool> &visited) {
    if (visited[i] == true) return;

    visited[i] = true;
    for (const auto &r : rooms[i]) {
        dfs(r, rooms, visited);
    }
}