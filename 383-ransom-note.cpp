// Solution 1 two vectors

bool canConstruct(string ransomNote, string magazine) {
    vector<char> ransom, maga;
    ransom.assign(ransomNote.begin(), ransomNote.end());
    maga.assign(magazine.begin(), magazine.end());

    sort(ransom.begin(), ransom.end());
    sort(maga.begin(), maga.end());

    auto rp = ransom.begin();

    for (auto mb = maga.begin(); mb != maga.end(); ++mb){
        if (rp == ransom.end()) return true;
        if (*rp == *mb) ++rp;
    }
    if (rp == ransom.end()) return true;
    else return false;
}

// Solution 2 hash table

bool canConstruct(string ransomNote, string magazine) {
    int *a = new int[26]();
    for (char c : magazine){
        ++a[c-'a'];
    }
    for (char c : ransomNote){
        if (a[c-'a'] == 0) return false;
        --a[c-'a'];
    }
    return true;
}