bool searchMatrix(vector<vector<int>>& matrix, int target) {
    int m_left = 0, m_right = matrix.size()-1;
    int n_left = 0, n_right = matrix[0].size()-1;
    while (m_left < m_right) {
        int m_mid = m_left + (m_right-m_left)/2;
        if (target > matrix[m_mid][n_right]) {
            m_left = m_mid + 1;
        }
        else {
            m_right = m_mid;
        }
    }
    int m = m_left;
    while (n_left < n_right) {
        int n_mid = n_left + (n_right-n_left)/2;
        if (target > matrix[m][n_mid]) {
            n_left = n_mid + 1;
        }
        else {
            n_right = n_mid;
        }
    }
    int n = n_left;
    return matrix[m][n] == target;
}