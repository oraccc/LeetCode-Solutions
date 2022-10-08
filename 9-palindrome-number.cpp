//Solution 1

bool isPalindrome(int x) {
    string s = to_string(x);
    return s == string(s.rbegin(), s.rend());
}

//Solution 2: get the left and right number from the remaining number

bool isPalindrome(int x) {
    if(x < 0) return false;

    int tmp = x;
    int multiply = 1;
    while (tmp / 10){
        multiply *= 10;
        tmp /= 10;
    }

    tmp = x;
    while (multiply > 1){
        int left = tmp / multiply;
        int right = tmp % 10;
        if (left != right) 
            return false;
        tmp = (tmp % multiply) / 10;
        multiply /= 100;
    }

    return true;
}