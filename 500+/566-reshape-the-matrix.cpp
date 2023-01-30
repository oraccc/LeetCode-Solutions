vector<vector<int>> matrixReshape(vector<vector<int>>& mat, int r, int c) {
    int m = mat.size(), n = mat[0].size();
    if (m*n != r*c) {
        return mat;
    }
    vector<vector<int>> ans(r ,vector<int>(c, 0));
    int curr_r = 0, curr_c = 0;
    for (int curr_m = 0; curr_m < m; ++curr_m) {
        for (int curr_n = 0; curr_n < n; ++curr_n) {
                ans[curr_r][curr_c] = mat[curr_m][curr_n];
                ++curr_c;
                if (curr_c == c) {
                    curr_c = 0;
                    ++curr_r;
                }
                
        }
    }
    return ans;
}