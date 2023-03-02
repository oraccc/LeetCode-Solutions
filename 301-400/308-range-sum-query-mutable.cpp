class NumArray {
    vector<int> st;
    int n;

    void buildTree(vector<int> &nums, int pos, int left, int right) {
        if (left == right) {
            st[pos] = nums[left];
            return;
        }
        int mid = left + (right-left)/2;
        buildTree(nums, 2*pos+1, left, mid);
        buildTree(nums, 2*pos+2, mid+1, right);
        st[pos] = st[2*pos+1] + st[2*pos+2];
    }

    void updateTree(int pos, int left, int right, int index, int val) {
        if(index < left || index > right) return;
        if(left == right){
            st[pos] = val;
            return;
        }

        int mid = left + (right-left)/2;
        updateTree(2*pos+1, left, mid, index, val);
        updateTree(2*pos+2, mid+1, right, index, val);
        st[pos] = st[2*pos+1] + st[2*pos+2];
    }

    int rangeTree(int qlow, int qhigh, int low, int high, int pos) {
        if (qlow <= low && qhigh >= high) {
            return st[pos];
        }
        if (qlow > high || qhigh < low) {
            return 0;
        }

        int mid = low + (high-low)/2;
        return (rangeTree(qlow, qhigh, low, mid, 2*pos+1) + rangeTree(qlow, qhigh, mid+1, high, 2*pos+2));
    }

public:
    NumArray(vector<int>& nums) {
        n = nums.size();
        st = vector<int>(4*n, 0);
        buildTree(nums, 0, 0, n-1);
        
    }
    
    void update(int index, int val) {
        updateTree(0, 0, n-1, index, val);
    }
    
    int sumRange(int left, int right) {
        return rangeTree(left, right, 0, n-1, 0);
    }
};

/**
 * Your NumArray object will be instantiated and called as such:
 * NumArray* obj = new NumArray(nums);
 * obj->update(index,val);
 * int param_2 = obj->sumRange(left,right);
 */