string longestCommonPrefix(vector<string>& strs) {
    string prefix = "";
    int max_length = 0;
    for (const auto &s : strs)
    {
        if (s.size() == 0) return prefix;
        if (s.size() > max_length) max_length = s.size();
    }

    for (int index = 0; index < max_length; ++index)
    {
        string first_prefix = strs[0].substr(0, index + 1);
        for (int i = 1; i < strs.size(); ++i)
        {
            if (strs[i].substr(0, index + 1) != first_prefix) return prefix;
        }

        prefix = first_prefix;
    }

    return prefix;
    
}