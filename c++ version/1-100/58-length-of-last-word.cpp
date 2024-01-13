int lengthOfLastWord(string s) {
    int lastCount = 0, currCount = 0;
    for (int i = 0; i < s.size(); ++i) {
        if (s[i] != ' ') {
            ++currCount;
        }
        else {
            if (currCount != 0) {
                lastCount = currCount;
                currCount = 0;
            }
        }
    }
    if (currCount != 0) return currCount;
    return lastCount;
    
}